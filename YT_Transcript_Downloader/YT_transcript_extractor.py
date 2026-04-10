"""
YouTube Transcript Extractor
============================
Extracts transcripts/subtitles from YouTube videos.

Compatible with youtube-transcript-api v1.2.0+

Usage:
    1. Set youtubeVideoLink below and run the script
    2. Or run with command line: python YT_transcript_extractor.py <url>
    3. Or run without arguments for interactive mode

Examples:
    python YT_transcript_extractor.py
    python YT_transcript_extractor.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
    python YT_transcript_extractor.py dQw4w9WgXcQ -o transcript.txt
"""

# ============================================================
# CONFIGURATION - Set your YouTube video link here
# ============================================================
youtubeVideoLink = "https://www.youtube.com/watch?v=Du-GWB4g7GY"  # <-- PASTE YOUR YOUTUBE LINK HERE
# Example: youtubeVideoLink = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Optional settings (only used when youtubeVideoLink is set above)
outputFile = None  # Set to filename like "transcript.txt" to save, or None for console
outputFormat = "text"  # Options: "text", "json", "srt"
language = "en"  # Language code: "en", "es", "fr", etc.
includeTimestamps = False  # Set to True to include timestamps in text output
# ============================================================

import re
import json
import argparse
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi

# Create API instance (required for v1.2.0+)
ytt_api = YouTubeTranscriptApi()


def extract_video_id(url_or_id: str) -> str:
    """
    Extract the video ID from various YouTube URL formats or return the ID if already provided.

    Supported formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/v/VIDEO_ID
        - VIDEO_ID (direct ID)
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'  # Direct video ID
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract video ID from: {url_or_id}")


def get_transcript(video_id: str, language: str = 'en') -> list:
    """
    Fetch the transcript for a given video ID.

    Args:
        video_id: YouTube video ID
        language: Language code (e.g., 'en', 'es', 'fr')

    Returns:
        List of transcript segments with 'text', 'start', and 'duration' keys
    """
    try:
        # Fetch transcript using the new API (v1.2.0+)
        # ytt_api.fetch() returns a FetchedTranscript object which is iterable
        fetched_transcript = ytt_api.fetch(video_id, languages=[language])

        # Convert to list of dicts for compatibility
        transcript = [
            {
                'text': snippet.text,
                'start': snippet.start,
                'duration': snippet.duration
            }
            for snippet in fetched_transcript
        ]
        return transcript

    except Exception as e:
        # If specified language not found, try without language filter
        try:
            print(f"Note: '{language}' transcript not found, fetching default...")
            fetched_transcript = ytt_api.fetch(video_id)

            transcript = [
                {
                    'text': snippet.text,
                    'start': snippet.start,
                    'duration': snippet.duration
                }
                for snippet in fetched_transcript
            ]
            return transcript

        except Exception as e2:
            raise Exception(f"Could not retrieve transcript: {e2}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm format for SRT."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_simple(seconds: float) -> str:
    """Convert seconds to MM:SS format for simple display."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def to_text(transcript: list, include_timestamps: bool = False) -> str:
    """Convert transcript to plain text format."""
    lines = []
    for segment in transcript:
        if include_timestamps:
            timestamp = format_timestamp_simple(segment['start'])
            lines.append(f"[{timestamp}] {segment['text']}")
        else:
            lines.append(segment['text'])
    return '\n'.join(lines)


def to_json(transcript: list) -> str:
    """Convert transcript to JSON format."""
    return json.dumps(transcript, indent=2, ensure_ascii=False)


def to_srt(transcript: list) -> str:
    """Convert transcript to SRT subtitle format."""
    srt_lines = []
    for i, segment in enumerate(transcript, 1):
        start = segment['start']
        end = start + segment['duration']

        srt_lines.append(str(i))
        srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        srt_lines.append(segment['text'])
        srt_lines.append('')  # Empty line between entries

    return '\n'.join(srt_lines)


def list_available_transcripts(video_id: str) -> None:
    """List all available transcripts for a video."""
    try:
        # Use new API method (v1.2.0+)
        transcript_list = ytt_api.list(video_id)

        print(f"\nAvailable transcripts for video: {video_id}")
        print("-" * 50)

        for transcript in transcript_list:
            info = f"  - {transcript.language} ({transcript.language_code})"
            if transcript.is_translatable:
                info += " [translatable]"
            if transcript.is_generated:
                info += " [auto-generated]"
            else:
                info += " [manual]"
            print(info)

        print()
    except Exception as e:
        print(f"Error listing transcripts: {e}")


def extract_transcript(
        url_or_id: str,
        output_file: Optional[str] = None,
        output_format: str = 'text',
        language: str = 'en',
        include_timestamps: bool = False,
        list_langs: bool = False
) -> Optional[str]:
    """
    Main function to extract transcript from a YouTube video.

    Args:
        url_or_id: YouTube URL or video ID
        output_file: Path to save output (None for console output)
        output_format: 'text', 'json', or 'srt'
        language: Language code
        include_timestamps: Include timestamps in text output
        list_langs: Just list available languages

    Returns:
        Formatted transcript string
    """
    # Extract video ID
    video_id = extract_video_id(url_or_id)
    print(f"Video ID: {video_id}")

    # If just listing languages
    if list_langs:
        list_available_transcripts(video_id)
        return None

    # Fetch transcript
    print(f"Fetching transcript in '{language}'...")
    transcript = get_transcript(video_id, language)
    print(f"Retrieved {len(transcript)} segments")

    # Format output
    if output_format == 'json':
        output = to_json(transcript)
    elif output_format == 'srt':
        output = to_srt(transcript)
    else:  # text
        output = to_text(transcript, include_timestamps)

    # Save or print
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Transcript saved to: {output_file}")
    else:
        print("\n" + "=" * 50)
        print("TRANSCRIPT")
        print("=" * 50 + "\n")
        print(output)

    return output


def interactive_mode():
    """
    Run the transcript extractor in interactive mode.
    Prompts user for YouTube video link and options.
    """
    print("\n" + "=" * 50)
    print("  YouTube Transcript Extractor - Interactive Mode")
    print("=" * 50 + "\n")

    # Get video link from user
    video_link = input("Enter YouTube video URL or ID: ").strip()

    if not video_link:
        print("Error: No video link provided.")
        return

    # Get optional settings
    print("\n--- Optional Settings (press Enter for defaults) ---")

    lang = input("Language code [en]: ").strip() or 'en'

    output_format = input("Output format (text/json/srt) [text]: ").strip().lower() or 'text'
    if output_format not in ['text', 'json', 'srt']:
        print(f"Invalid format '{output_format}', using 'text'")
        output_format = 'text'

    output_file = input("Output file path (leave empty for console): ").strip() or None

    timestamps_input = input("Include timestamps? (y/n) [n]: ").strip().lower()
    include_timestamps = timestamps_input in ['y', 'yes']

    list_langs_input = input("List available languages first? (y/n) [n]: ").strip().lower()
    list_langs = list_langs_input in ['y', 'yes']

    print()

    # Extract transcript
    try:
        extract_transcript(
            url_or_id=video_link,
            output_file=output_file,
            output_format=output_format,
            language=lang,
            include_timestamps=include_timestamps,
            list_langs=list_langs
        )
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract transcripts from YouTube videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Uses youtubeVideoLink from config or interactive mode
  %(prog)s https://www.youtube.com/watch?v=VIDEO_ID
  %(prog)s VIDEO_ID -o transcript.txt
  %(prog)s VIDEO_ID -f srt -o subtitles.srt
  %(prog)s VIDEO_ID -l es --timestamps
  %(prog)s VIDEO_ID --list-languages
        """
    )

    parser.add_argument(
        'video',
        nargs='?',  # Makes it optional
        default=None,
        help='YouTube URL or video ID (if not provided, uses config or interactive mode)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: print to console)'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['text', 'json', 'srt'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '-l', '--language',
        default='en',
        help='Language code (default: en)'
    )
    parser.add_argument(
        '--timestamps',
        action='store_true',
        help='Include timestamps in text output'
    )
    parser.add_argument(
        '--list-languages',
        action='store_true',
        help='List available transcript languages'
    )

    args = parser.parse_args()

    # Priority: 1. Command line arg  2. Config variable  3. Interactive mode
    if args.video is not None:
        # Use command line argument
        video_link = args.video
        out_file = args.output
        out_format = args.format
        lang = args.language
        timestamps = args.timestamps
        list_langs = args.list_languages
    elif youtubeVideoLink:
        # Use the hardcoded config variable
        print("Using video link from configuration...")
        video_link = youtubeVideoLink
        out_file = outputFile
        out_format = outputFormat
        lang = language
        timestamps = includeTimestamps
        list_langs = False
    else:
        # No video provided, run interactive mode
        interactive_mode()
        return

    try:
        extract_transcript(
            url_or_id=video_link,
            output_file=out_file,
            output_format=out_format,
            language=lang,
            include_timestamps=timestamps,
            list_langs=list_langs
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()


# """
# YouTube Transcript Extractor
# ============================
# Extracts transcripts/subtitles from YouTube videos.
#
# Usage:
#     1. Set youtubeVideoLink below and run the script
#     2. Or run with command line: python YT_transcript_extractor.py <url>
#     3. Or run without arguments for interactive mode
#
# Examples:
#     python YT_transcript_extractor.py
#     python YT_transcript_extractor.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
#     python YT_transcript_extractor.py dQw4w9WgXcQ -o transcript.txt
# """
#
# # ============================================================
# # CONFIGURATION - Set your YouTube video link here
# # ============================================================
# youtubeVideoLink = "https://www.youtube.com/watch?v=rqk0baXEaL8"  # <-- PASTE YOUR YOUTUBE LINK HERE
# # Example: youtubeVideoLink = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
#
# # Optional settings (only used when youtubeVideoLink is set above)
# outputFile = None  # Set to filename like "transcript.txt" to save, or None for console
# outputFormat = "text"  # Options: "text", "json", "srt"
# language = "en"  # Language code: "en", "es", "fr", etc.
# includeTimestamps = False  # Set to True to include timestamps in text output
# # ============================================================
#
# import re
# import json
# import argparse
# from typing import Optional
#
# from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api._errors import (
#     TranscriptsDisabled,
#     NoTranscriptFound,
#     VideoUnavailable,
#     CouldNotRetrieveTranscript  # This is the correct exception (not NoTranscriptAvailable)
# )
#
#
# def extract_video_id(url_or_id: str) -> str:
#     """
#     Extract the video ID from various YouTube URL formats or return the ID if already provided.
#
#     Supported formats:
#         - https://www.youtube.com/watch?v=VIDEO_ID
#         - https://youtu.be/VIDEO_ID
#         - https://www.youtube.com/embed/VIDEO_ID
#         - https://www.youtube.com/v/VIDEO_ID
#         - VIDEO_ID (direct ID)
#     """
#     patterns = [
#         r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
#         r'^([a-zA-Z0-9_-]{11})$'  # Direct video ID
#     ]
#
#     for pattern in patterns:
#         match = re.search(pattern, url_or_id)
#         if match:
#             return match.group(1)
#
#     raise ValueError(f"Could not extract video ID from: {url_or_id}")
#
#
# def get_transcript(video_id: str, language: str = 'en') -> list:
#     """
#     Fetch the transcript for a given video ID.
#
#     Args:
#         video_id: YouTube video ID
#         language: Language code (e.g., 'en', 'es', 'fr')
#
#     Returns:
#         List of transcript segments with 'text', 'start', and 'duration' keys
#     """
#     try:
#         # Try to get transcript in the specified language
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#
#         try:
#             # First try to find manually created transcript
#             transcript = transcript_list.find_manually_created_transcript([language])
#         except NoTranscriptFound:
#             try:
#                 # Fall back to auto-generated transcript
#                 transcript = transcript_list.find_generated_transcript([language])
#             except NoTranscriptFound:
#                 # Try to get any available transcript and translate it
#                 available = list(transcript_list)
#                 if available:
#                     transcript = available[0]
#                     if language != transcript.language_code:
#                         print(f"Note: Translating from {transcript.language_code} to {language}")
#                         transcript = transcript.translate(language)
#                 else:
#                     raise CouldNotRetrieveTranscript(video_id)
#
#         return transcript.fetch()
#
#     except TranscriptsDisabled:
#         raise Exception(f"Transcripts are disabled for video: {video_id}")
#     except VideoUnavailable:
#         raise Exception(f"Video unavailable: {video_id}")
#     except CouldNotRetrieveTranscript:
#         raise Exception(f"Could not retrieve transcript for video: {video_id}")
#
#
# def format_timestamp(seconds: float) -> str:
#     """Convert seconds to HH:MM:SS,mmm format for SRT."""
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     secs = int(seconds % 60)
#     millis = int((seconds % 1) * 1000)
#     return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
#
#
# def format_timestamp_simple(seconds: float) -> str:
#     """Convert seconds to MM:SS format for simple display."""
#     minutes = int(seconds // 60)
#     secs = int(seconds % 60)
#     return f"{minutes:02d}:{secs:02d}"
#
#
# def to_text(transcript: list, include_timestamps: bool = False) -> str:
#     """Convert transcript to plain text format."""
#     lines = []
#     for segment in transcript:
#         if include_timestamps:
#             timestamp = format_timestamp_simple(segment['start'])
#             lines.append(f"[{timestamp}] {segment['text']}")
#         else:
#             lines.append(segment['text'])
#     return '\n'.join(lines)
#
#
# def to_json(transcript: list) -> str:
#     """Convert transcript to JSON format."""
#     return json.dumps(transcript, indent=2, ensure_ascii=False)
#
#
# def to_srt(transcript: list) -> str:
#     """Convert transcript to SRT subtitle format."""
#     srt_lines = []
#     for i, segment in enumerate(transcript, 1):
#         start = segment['start']
#         end = start + segment['duration']
#
#         srt_lines.append(str(i))
#         srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
#         srt_lines.append(segment['text'])
#         srt_lines.append('')  # Empty line between entries
#
#     return '\n'.join(srt_lines)
#
#
# def list_available_transcripts(video_id: str) -> None:
#     """List all available transcripts for a video."""
#     try:
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#
#         print(f"\nAvailable transcripts for video: {video_id}")
#         print("-" * 50)
#
#         manual = []
#         generated = []
#
#         for transcript in transcript_list:
#             info = f"  - {transcript.language} ({transcript.language_code})"
#             if transcript.is_translatable:
#                 info += " [translatable]"
#
#             if transcript.is_generated:
#                 generated.append(info)
#             else:
#                 manual.append(info)
#
#         if manual:
#             print("\nManually created:")
#             for t in manual:
#                 print(t)
#
#         if generated:
#             print("\nAuto-generated:")
#             for t in generated:
#                 print(t)
#
#         print()
#     except Exception as e:
#         print(f"Error listing transcripts: {e}")
#
#
# def extract_transcript(
#         url_or_id: str,
#         output_file: Optional[str] = None,
#         output_format: str = 'text',
#         language: str = 'en',
#         include_timestamps: bool = False,
#         list_langs: bool = False
# ) -> Optional[str]:
#     """
#     Main function to extract transcript from a YouTube video.
#
#     Args:
#         url_or_id: YouTube URL or video ID
#         output_file: Path to save output (None for console output)
#         output_format: 'text', 'json', or 'srt'
#         language: Language code
#         include_timestamps: Include timestamps in text output
#         list_langs: Just list available languages
#
#     Returns:
#         Formatted transcript string
#     """
#     # Extract video ID
#     video_id = extract_video_id(url_or_id)
#     print(f"Video ID: {video_id}")
#
#     # If just listing languages
#     if list_langs:
#         list_available_transcripts(video_id)
#         return None
#
#     # Fetch transcript
#     print(f"Fetching transcript in '{language}'...")
#     transcript = get_transcript(video_id, language)
#     print(f"Retrieved {len(transcript)} segments")
#
#     # Format output
#     if output_format == 'json':
#         output = to_json(transcript)
#     elif output_format == 'srt':
#         output = to_srt(transcript)
#     else:  # text
#         output = to_text(transcript, include_timestamps)
#
#     # Save or print
#     if output_file:
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(output)
#         print(f"Transcript saved to: {output_file}")
#     else:
#         print("\n" + "=" * 50)
#         print("TRANSCRIPT")
#         print("=" * 50 + "\n")
#         print(output)
#
#     return output
#
#
# def interactive_mode():
#     """
#     Run the transcript extractor in interactive mode.
#     Prompts user for YouTube video link and options.
#     """
#     print("\n" + "=" * 50)
#     print("  YouTube Transcript Extractor - Interactive Mode")
#     print("=" * 50 + "\n")
#
#     # Get video link from user
#     youtubeVideoLink = input("Enter YouTube video URL or ID: ").strip()
#
#     if not youtubeVideoLink:
#         print("Error: No video link provided.")
#         return
#
#     # Get optional settings
#     print("\n--- Optional Settings (press Enter for defaults) ---")
#
#     language = input("Language code [en]: ").strip() or 'en'
#
#     output_format = input("Output format (text/json/srt) [text]: ").strip().lower() or 'text'
#     if output_format not in ['text', 'json', 'srt']:
#         print(f"Invalid format '{output_format}', using 'text'")
#         output_format = 'text'
#
#     output_file = input("Output file path (leave empty for console): ").strip() or None
#
#     timestamps_input = input("Include timestamps? (y/n) [n]: ").strip().lower()
#     include_timestamps = timestamps_input in ['y', 'yes']
#
#     list_langs_input = input("List available languages only? (y/n) [n]: ").strip().lower()
#     list_langs = list_langs_input in ['y', 'yes']
#
#     print()
#
#     # Extract transcript
#     try:
#         extract_transcript(
#             url_or_id=youtubeVideoLink,
#             output_file=output_file,
#             output_format=output_format,
#             language=language,
#             include_timestamps=include_timestamps,
#             list_langs=list_langs
#         )
#     except Exception as e:
#         print(f"Error: {e}")
#
#
# def main():
#     parser = argparse.ArgumentParser(
#         description='Extract transcripts from YouTube videos',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   %(prog)s                                    # Uses youtubeVideoLink from config or interactive mode
#   %(prog)s https://www.youtube.com/watch?v=VIDEO_ID
#   %(prog)s VIDEO_ID -o transcript.txt
#   %(prog)s VIDEO_ID -f srt -o subtitles.srt
#   %(prog)s VIDEO_ID -l es --timestamps
#   %(prog)s VIDEO_ID --list-languages
#         """
#     )
#
#     parser.add_argument(
#         'video',
#         nargs='?',  # Makes it optional
#         default=None,
#         help='YouTube URL or video ID (if not provided, uses config or interactive mode)'
#     )
#     parser.add_argument(
#         '-o', '--output',
#         help='Output file path (default: print to console)'
#     )
#     parser.add_argument(
#         '-f', '--format',
#         choices=['text', 'json', 'srt'],
#         default='text',
#         help='Output format (default: text)'
#     )
#     parser.add_argument(
#         '-l', '--language',
#         default='en',
#         help='Language code (default: en)'
#     )
#     parser.add_argument(
#         '--timestamps',
#         action='store_true',
#         help='Include timestamps in text output'
#     )
#     parser.add_argument(
#         '--list-languages',
#         action='store_true',
#         help='List available transcript languages'
#     )
#
#     args = parser.parse_args()
#
#     # Priority: 1. Command line arg  2. Config variable  3. Interactive mode
#     if args.video is not None:
#         # Use command line argument
#         video_link = args.video
#         out_file = args.output
#         out_format = args.format
#         lang = args.language
#         timestamps = args.timestamps
#         list_langs = args.list_languages
#     elif youtubeVideoLink:
#         # Use the hardcoded config variable
#         print("Using video link from configuration...")
#         video_link = youtubeVideoLink
#         out_file = outputFile
#         out_format = outputFormat
#         lang = language
#         timestamps = includeTimestamps
#         list_langs = False
#     else:
#         # No video provided, run interactive mode
#         interactive_mode()
#         return
#
#     try:
#         extract_transcript(
#             url_or_id=video_link,
#             output_file=out_file,
#             output_format=out_format,
#             language=lang,
#             include_timestamps=timestamps,
#             list_langs=list_langs
#         )
#     except Exception as e:
#         print(f"Error: {e}")
#         exit(1)
#
#
# if __name__ == '__main__':
#     main()
