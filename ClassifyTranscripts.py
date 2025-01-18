'''This script performs classification, topic modelling, and Named Entity recognition on our transcripts '''

import Watchmen
import sys


def main():
    if len(sys.argv) < 1:
        exit("You must pass the OpenAI API key for this script to run.\nProcess stopped.")

    openai_api_key_path = sys.argv[1]

    w = Watchmen.Watchmen(
        openai_api_key_path=openai_api_key_path, include_youtube_explorer=False)
    w.build_entities_for_all()
    w.classify_transcripts()


if __name__ == '__main__':
    try:
        main()
    except:
        exit('Something happened. You should probably look into that.')
