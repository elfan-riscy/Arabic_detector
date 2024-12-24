import 'package:flutter/material.dart';

class HowToUsePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Cara Menggunakan'),
        backgroundColor: Colors.green[900],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView(
          children: [
            // Langkah-langkah penggunaan
            Text(
              '1. Masukkan teks bahasa Arab.\n2. Tekan tombol Identifikasi.\n3. Gunakan fitur Scan Foto untuk mendeteksi dari gambar.',
              style: TextStyle(fontSize: 18),
            ),
            SizedBox(height: 20), // Jarak antar bagian

            // Judul untuk penjelasan rule Ilmu Nahwu
            Text(
              'A. Rule Ilmu Nahwu',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 10),

            // Penjelasan rule Ilmu Nahwu
            Text(
              'Berikut penjelasan mengenai rule ilmu nahwu:\n\n'
                  '1) Kalimat Isim (kata benda) adalah setiap kata yang mempunyai arti benda, baik konkret maupun abstrak, tanpa ada unsur waktu di dalamnya. Isim memiliki ciri khusus yang membedakannya dengan jenis kata lain. Cirinya antara lain:\n'
                  '- Diawali dengan al (ال). Contoh: الكتاب\n'
                  '- Diakhiri dengan tanwin. Contoh: كتابٌ\n'
                  '- Diawali salah satu huruf jar: من، الى، عن، على، في، رب، الباء، الكاف، الام، الواو، الباء، التأ\n'
                  '- Diawali huruf nida. Contoh: يا ادم\n'
                  '- Disandarkan (diidlofatkan) kepada kata benda lainnya, kebanyakan untuk menunjukkan pemilikan. Contoh: كتاب الله\n\n'
                  '2) Kalimat Fi’il (kata kerja) adalah kata yang menunjukkan suatu pekerjaan yang berkaitan dengan waktu, baik lampau, sekarang, maupun yang akan datang. Kata kerja dalam bahasa Arab dibedakan menjadi tiga macam, yaitu: fi’il madi (lampau), fi’il mudari’ (sekarang dan akan datang), dan fi’il amr (perintah). Ciri khas dari fi’il adalah:\n'
                  '- Contoh: لينص\n'
                  '- Diawali salah satu huruf mudlori’ (ن، أ، ت، ي)\n'
                  '- Diawali dengan س atau سوف\n'
                  '- Diakhiri dengan Ta’ taknis (ت). Contoh: نصرت\n'
                  '- Diakhiri dengan Ta’ fa’il.\n\n'
                  '3) Kalimat Harf/Huruf adalah setiap kata yang tidak memiliki makna kecuali menyertainya dengan kata lain. Harf atau huruf dipandang sebagai kata tugas atau kata penghubung, biasanya terdiri kurang dari tiga huruf.',
              style: TextStyle(fontSize: 18),
              textAlign: TextAlign.justify,
            ),
          ],
        ),
      ),
    );
  }
}
