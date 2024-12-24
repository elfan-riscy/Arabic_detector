import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../models/arabic_detector.dart';
import 'about_page.dart';
import 'how_to_use_page.dart';
import 'dart:io';

class DetectionPage extends StatefulWidget {
  @override
  _DetectionPageState createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  final TextEditingController _textController = TextEditingController();
  final ArabicDetector _detector = ArabicDetector();
  String _type = ''; // Jenis kalimat
  String _features = ''; // Ciri-ciri kalimat
  File? _selectedImage; // File gambar yang diunggah

  void _detectSentence() async {
    if (_textController.text.isNotEmpty) {
      try {
        final result = await _detector.predict(_textController.text);
        // Debugging untuk memeriksa hasil dari model
        print('Result: $result');
        setState(() {
          _type = (result['type']?.toString() ?? 'Tidak Diketahui');
          _features = (result['features']?.toString() ?? '');
        });
      } catch (e) {
        setState(() {
          _type = 'Error';
          _features = 'Terjadi kesalahan saat mendeteksi.';
        });
      }
    } else {
      setState(() {
        _type = 'Masukkan teks terlebih dahulu.';
        _features = '';
      });
    }
  }

  Future<void> _pickImageFromGallery() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
      });

      try {
        final result = await _detector.predictFromImage(_selectedImage!);
        print('Image Result: $result');
        setState(() {
          _type = (result['type']?.toString() ?? 'Tidak Diketahui');
          _features = (result['features']?.toString() ?? '');
        });
      } catch (e) {
        setState(() {
          _type = 'Error';
          _features = 'Terjadi kesalahan saat mendeteksi gambar.';
        });
      }
    }
  }

  Future<void> _pickImageFromCamera() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.camera);

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
      });

      try {
        final result = await _detector.predictFromImage(_selectedImage!);
        print('Camera Result: $result');
        setState(() {
          _type = (result['type']?.toString() ?? 'Tidak Diketahui');
          _features = (result['features']?.toString() ?? '');
        });
      } catch (e) {
        setState(() {
          _type = 'Error';
          _features = 'Terjadi kesalahan saat mendeteksi gambar.';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Deteksi Kalimat'),
        backgroundColor: Colors.green[900],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextField(
                controller: _textController,
                decoration: InputDecoration(
                  hintText: 'Masukkan kalimat bahasa Arab...',
                  border: OutlineInputBorder(),
                  fillColor: Colors.white,
                  filled: true,
                ),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 16),
              ElevatedButton(
                onPressed: _detectSentence,
                child: Text('Identifikasi'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green[900],
                  foregroundColor: Colors.white,
                  padding: EdgeInsets.symmetric(vertical: 15),
                ),
              ),
              SizedBox(height: 16),
              ElevatedButton(
                onPressed: _pickImageFromGallery,
                child: Text('Unggah Gambar'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue[900],
                  foregroundColor: Colors.white,
                  padding: EdgeInsets.symmetric(vertical: 15),
                ),
              ),
              SizedBox(height: 16),
              if (_selectedImage != null)
                Column(
                  children: [
                    Text(
                      'Gambar yang Dipilih:',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                    SizedBox(height: 8),
                    Image.file(
                      _selectedImage!,
                      height: 200,
                      fit: BoxFit.cover,
                    ),
                  ],
                ),
              if (_type.isNotEmpty)
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Hasil Deteksi:',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Jenis Kalimat: $_type',
                      style: TextStyle(fontSize: 16, color: Colors.black),
                    ),
                    SizedBox(height: 4),
                    Text(
                      'Ciri-ciri: $_features',
                      style: TextStyle(fontSize: 16, color: Colors.black),
                    ),
                  ],
                ),
            ],
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _pickImageFromCamera, // Fungsi untuk membuka kamera
        child: Icon(Icons.camera_alt),
        backgroundColor: Colors.green[900],
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      bottomNavigationBar: BottomAppBar(
        shape: CircularNotchedRectangle(),
        notchMargin: 8.0,
        color: Colors.green[900],
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            IconButton(
              icon: Icon(Icons.info, color: Colors.white),
              onPressed: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => AboutPage()),
              ),
            ),
            IconButton(
              icon: Icon(Icons.help, color: Colors.white),
              onPressed: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => HowToUsePage()),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
