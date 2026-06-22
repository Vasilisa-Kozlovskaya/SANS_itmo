import os
from warcio.archiveiterator import ArchiveIterator

def convert_warc_to_text(warc_path: str, output_path: str):
    """Извлекает сырой контент из WARC файла и сохраняет в .txt."""
    with open(warc_path, 'rb') as stream:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for record in ArchiveIterator(stream):
                # Нас интересуют только ответы серверов (responses)
                if record.rec_type == 'response':
                    # Извлекаем содержимое
                    payload = record.content_stream().read()
                    try:
                        # Декодируем, игнорируя ошибки (бинарные данные)
                        text = payload.decode('utf-8', errors='ignore')
                        if text.strip():
                            out_f.write(text + "\n" + "="*50 + "\n")
                    except Exception as e:
                        continue
    print(f"Конвертация завершена. Файл сохранен в: {output_path}")