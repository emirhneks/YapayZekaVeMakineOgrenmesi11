from openai import OpenAI
from dotenv import load_dotenv
import os

# .env dosyasından güvenli API anahtarı yüklemesi
load_dotenv()

# OpenAI istemcisini çevresel değişkenlerden alınan API anahtarı ile başlat
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def complete_text(prompt):
    """
    OpenAI'nin GPT modelini kullanarak metin tamamlama işlemi yapar.

    Parametreler:
        prompt (str): Tamamlanacak giriş metni

    Dönüş:
        str: Yapay zeka tarafından oluşturulan metin
    """
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=200,
        temperature=0.8,
        stop=None,
    )

    return response.choices[0].text.strip()


def main():
    """
    Metin tamamlama arayüzünü yöneten ana fonksiyon.
    Kullanıcıların metin girişi yapmasını ve yapay zeka
    tarafından tamamlanan metinleri almalarını sağlar.
    """
    print("Metin tamamlama uygulamasına hoş geldiniz!")
    print("Çıkmak için 'çıkış' yazabilirsiniz.\n")

    while True:
        prompt = input("Lütfen bir metin girin: ")

        if prompt.lower() == "çıkış":
            print("Uygulamadan çıkılıyor...")
            break

        completed_text = complete_text(prompt)
        print("\nTamamlanan Metin:")
        print(f"{prompt} {completed_text}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
