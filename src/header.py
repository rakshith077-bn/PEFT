from pyfiglet import Figlet, figlet_format

def welcome_note():
    text = figlet_format("\n PeftKIT \n", font="ansi_shadow")
    colored_text = f"\033[31m{text}\033[0m" 
    print(colored_text)
    print(" ")
    print("- python3 finetune.py --help' for help with finetune.py \n")
    print("- python3 feature_extraction.py --help for help with feature extraction")

def main_intro():
    text = figlet_format("Developed by Rakshith", font="standard")
    colored_text = f"\033[31m{text}\033[0m"
    print(colored_text)

def general_info():
    text = figlet_format("Welcome", font="ansi_shadow")
    colored_text = f"\033[31m{text}\033[0m"
    print(colored_text)

def data_load():
    f = Figlet(font='ntgreek', justify="center")
    print(f.renderText('Dataset Loaded Successfully'))
