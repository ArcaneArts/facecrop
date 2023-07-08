import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:args/command_runner.dart';
import 'package:fast_log/fast_log.dart';
import 'package:googleapis/connectors/v1.dart';
import 'package:googleapis/vision/v1.dart';
import 'package:googleapis_auth/googleapis_auth.dart';
import 'package:googleapis_auth/auth_io.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:path/path.dart';

late CommandRunner _runner;
String? jsonFile;
String? input;
String? output;

void main(List<String> args) {
  execute(args);
}

Future<void> execute(List<String> args) async {
  Reactor<void> reactor = Reactor<void>();
  _runner = CommandRunner("facecrop", "A face cropper");
  _runner.argParser
      .addOption("json", abbr: "j", callback: (k) => jsonFile = k ?? jsonFile);
  _runner.argParser
      .addOption("input", abbr: "i", callback: (k) => input = k ?? input);
  _runner.argParser
      .addOption("output", abbr: "o", callback: (k) => output = k ?? output);
  _runner.parse(args);

  final jsonString = await File(jsonFile!).readAsString();
  final jsonKey = json.decode(jsonString);
  final credentials = ServiceAccountCredentials.fromJson(jsonKey);
  final scopes = [VisionApi.cloudVisionScope];
  List<AutoRefreshingAuthClient> clients = [];

  Future<AutoRefreshingAuthClient> _createClient() async {
    AutoRefreshingAuthClient c =
        await clientViaServiceAccount(credentials, scopes);
    clients.add(c);
    return c;
  }

  Future<void> faceCrop(File input, File output, int minX, int maxX, int minY,
      int maxY, double maxPaddingPercent) async {
    img.Image? image = img.decodeImage(await input.readAsBytes());

    if (image == null) {
      print("Failed to decode ${input.path}");
      return;
    }

    // Calculate padding
    double paddingX = (maxX - minX) * maxPaddingPercent;
    double paddingY = (maxY - minY) * maxPaddingPercent;

    // Ensure the cropping area is within the image
    int cropX = (minX - paddingX).clamp(0, image.width.toDouble()).toInt();
    int cropY = (minY - paddingY).clamp(0, image.height.toDouble()).toInt();

    // Ensure the end coordinates (including padding) are within the image
    int endX = (maxX + paddingX).clamp(0, image.width.toDouble()).toInt();
    int endY = (maxY + paddingY).clamp(0, image.height.toDouble()).toInt();

    // Calculate the crop width and height
    int cropW = endX - cropX;
    int cropH = endY - cropY;

    // Save the cropped image
    await File(output.path).writeAsBytes(img.encodeJpg(img.copyResizeCropSquare(
        img.copyCrop(
          image,
          y: cropY,
          width: max(cropW, cropH),
          height: max(cropW, cropH),
          x: cropX,
        ),
        size: 512,
        antialias: true,
        interpolation: img.Interpolation.cubic)));
    success("Completed ${output.path}");
  }

  Future<Map<File, FaceAnnotation>> _completeBatch(List<File> files) async =>
      VisionApi(await _createClient())
          .images
          .annotate(BatchAnnotateImagesRequest(requests: [
            ...files.map((e) {
              return AnnotateImageRequest(
                image: Image()..contentAsBytes = e.readAsBytesSync(),
                features: [
                  Feature(
                    type: "FACE_DETECTION",
                  ),
                ],
              );
            })
          ]))
          .then((value) => value.responses!)
          .then((responses) {
        Map<File, FaceAnnotation> faceAnnotations = {};
        for (int i = 0; i < responses.length; i++) {
          File file = files[i];
          AnnotateImageResponse response = responses[i];

          if (response.faceAnnotations?.length == 1) {
            faceAnnotations[file] = response.faceAnnotations!.first;
          } else if (response.faceAnnotations != null) {
            response.faceAnnotations?.sort((a, b) =>
                (b.detectionConfidence ?? 0)
                    .compareTo(a.detectionConfidence ?? 0));
            faceAnnotations[file] = response.faceAnnotations!.first;
            warn(
                "${file.path} has ${response.faceAnnotations?.length} faces. Picking the highest confidence one. (${faceAnnotations[file]?.detectionConfidence ?? 0} of ${response.faceAnnotations?.map((e) => e.detectionConfidence ?? 0)})");
          } else {
            warn("${file.path} has no faces");
          }
        }

        success("Completed batch of ${files.length} files.");

        for (File file in faceAnnotations.keys) {
          FaceAnnotation a = faceAnnotations[file]!;
          double blurredFace = of(a.blurredLikelihood!).get();
          double headwear = of(a.headwearLikelihood!).get();
          double joy = of(a.joyLikelihood!).get();
          double sorrow = of(a.sorrowLikelihood!).get();
          double anger = of(a.angerLikelihood!).get();
          double surprise = of(a.surpriseLikelihood!).get();
          double underExposed = of(a.underExposedLikelihood!).get();
          double confidence = a.detectionConfidence ?? 0;
          double landmarkConfidence = a.landmarkingConfidence ?? 0;

          verbose(
              "IMG: ${file.path} CONFIDENCE: $confidence landmarkConfidence $landmarkConfidence blurredFace $blurredFace headwear $headwear joy $joy sorrow $sorrow anger $anger surprise $surprise underExposed $underExposed ");

          if (blurredFace > 0.2) {
            warn("Skipping ${file.path} because it may be blurred.");
            continue;
          }

          if (underExposed > 0.2) {
            warn("Skipping ${file.path} because it may be underexposed.");
            continue;
          }

          if (landmarkConfidence <= 0.4) {
            warn("Skipping ${file.path} because it may not be a full face.");
            continue;
          }

          if (confidence < 0.8) {
            warn("Skipping ${file.path} because it may not be a face.");
            continue;
          }

          int minX = 100000;
          int maxX = 0;
          int minY = 100000;
          int maxY = 0;

          for (Vertex i in a.boundingPoly?.vertices ?? []) {
            minX = min(minX, i.x ?? 0);
            maxX = max(maxX, i.x ?? 0);
            minY = min(minY, i.y ?? 0);
            maxY = max(maxY, i.y ?? 0);
          }

          reactor.add(() => faceCrop(
              file,
              File(
                join(output!, file.path.split(Platform.pathSeparator).last),
              ),
              minX,
              maxX,
              minY,
              maxY,
              0.13));
        }
        return faceAnnotations;
      });

  List<List<File>> _chunkFiles(List<File> allFiles, int mb) {
    List<List<File>> chunks = [];
    int maxBytes = mb * 1024 * 1024; // 32mb
    Map<File, FaceAnnotation> faceAnnotations = {};
    List<File> buffer = [];
    int currentBytes = 0;
    for (int i = 0; i < allFiles.length; i++) {
      File file = allFiles[i];
      int size = file.lengthSync();
      if (currentBytes + size > maxBytes) {
        verbose(
            "Batch ${chunks.length + 1} of ${buffer.length} files (${(currentBytes / 1024 / 1024).round()}mb)");
        chunks.add(buffer.toList());
        buffer = [];
        currentBytes = 0;
      }
      buffer.add(file);
      currentBytes += size;
    }
    if (buffer.isNotEmpty) {
      verbose(
          "Batch ${chunks.length + 1} of ${buffer.length} files (${(currentBytes / 1024 / 1024).round()}mb)");
      chunks.add(buffer.toList());
    }
    return chunks;
  }

  try {
    List<File> files = Directory(input!)
        .listSync(recursive: false, followLinks: false)
        .whereType<File>()
        .toList();
    verbose("Analyzing ${files.length} files");
    Map<File, FaceAnnotation> faceAnnotations = {};
    for (List<File> i in _chunkFiles(files, 16)) {
      reactor.add(() =>
          _completeBatch(i).then((value) => faceAnnotations.addAll(value)));
    }

    await reactor.run();
    success(
        "Analysis complete. Got ${faceAnnotations.length} singular face images.");
  } catch (e) {
    print('Failed to make authenticated request: $e');
  } finally {
    for (var element in clients) {
      element.close();
    }
  }
}

typedef ValueG<T> = Future<T> Function();

class Reactor<T> {
  List<ValueG<T>> queue = [];
  List<Future> core = [];
  int completed = 0;
  int maxFutures = 8;

  Reactor({this.maxFutures = 8});

  void add(ValueG<T> job) {
    queue.add(job);
  }

  Future<void> run() async {
    while (_tick()) {
      await Future.delayed(Duration(milliseconds: 3000));
      verbose("Working: ${queue.length + core.length} tasks left.");
    }
  }

  bool _tick() {
    while (core.length < maxFutures && queue.isNotEmpty) {
      completed++;
      Future<T> job = queue.removeAt(0)();
      core.add(job);
      job.then((value) => core.remove(job));
    }

    if (queue.isEmpty && core.isEmpty) {
      return false;
    }

    return true;
  }
}

enum Likelyhood {
  UNKNOWN,
  VERY_UNLIKELY,
  UNLIKELY,
  POSSIBLE,
  LIKELY,
  VERY_LIKELY
}

Likelyhood of(String s) =>
    Likelyhood.values.firstWhere((e) => e.toString().endsWith(s));

extension XLikelyhood on Likelyhood {
  double get() {
    switch (this) {
      case Likelyhood.UNKNOWN:
        return 0;
      case Likelyhood.VERY_UNLIKELY:
        return 0.2;
      case Likelyhood.UNLIKELY:
        return 0.4;
      case Likelyhood.POSSIBLE:
        return 0.6;
      case Likelyhood.LIKELY:
        return 0.8;
      case Likelyhood.VERY_LIKELY:
        return 1;
    }
  }
}

class Action {
  final int someProperty;
  final String someName;

  Action(this.someProperty, this.someName);

  void call() {
    // do the action
  }
}
