from art import text2art

def show_logo():
    logo = text2art("IFLRY TTS")
    print(logo)


def show_splash():
    print("============================================================")
    print("                      IFLRY TTS System                      ")
    print("      Powered by OuteTTS, Whisper, and MarianMT Models      ")
    print("============================================================\n")
