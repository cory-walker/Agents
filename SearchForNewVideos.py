import Watchmen
import sys


def main():
    if len(sys.argv) < 1:
        exit("You must pass the YouTube API key for this script to run.\nProcess stopped.")

    youtube_api_key_path = sys.argv[1]

    w = Watchmen.Watchmen(
        youtube_api_key_path=youtube_api_key_path, include_scholar=False)
    w.fetch_new_transcripts_for_channels()


if __name__ == '__main__':
    try:
        main()
    except:
        exit('Something happened. You should probably look into that.')
