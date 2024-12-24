import 'dart:io';
import 'package:flutter/services.dart';

class ArabicDetector {
  static const _channel = MethodChannel('arabic_sentence_model');

  Future<Map<String, String>> predict(String text) async {
    try {
      final result = await _channel.invokeMethod('predict', {'text': text});
      return Map<String, String>.from(result);
    } catch (e) {
      throw 'Error during prediction: $e';
    }
  }

  Future<Map<String, String>> predictFromImage(File image) async {
    try {
      final result = await _channel.invokeMethod('predictFromImage', {'imagePath': image.path});
      return Map<String, String>.from(result);
    } catch (e) {
      throw 'Error during image prediction: $e';
    }
  }
}
