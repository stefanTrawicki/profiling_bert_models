import argparse, subprocess, time, os, threading

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

import bert_imdb
# import bert_rotten_tomatoes
# import bert_ag_news
# import bert_snli
# import bert_yelp
# import roberta_imdb
# import roberta_rotten_tomatoes
# import roberta_ag_news
# import roberta_snli
# import roberta_yelp
# import finbert

models = [
    bert_imdb,
    # bert_rotten_tomatoes,
    # bert_ag_news,
    # bert_snli,
    # bert_yelp,
    # roberta_imdb,
    # roberta_rotten_tomatoes,
    # roberta_ag_news,
    # roberta_snli,
    # roberta_yelp,
    # finbert
]

parser = argparse.ArgumentParser(description='Runs a passed set of phrases on passed models, using DeepRecon to attack the model')

parser.add_argument("--phrases",
                    dest="phrases",
                    type=str,
                    default="phrases.txt",
                    help="A text file containing the phrases to run.")

parser.add_argument("--overall_runs",
                    dest="overall_runs",
                    type=int,
                    default=1,
                    help="How many overall runs to do. Each 'inference' will be as many rows as there are in the phrases input.")

parser.add_argument("--output_zip",
                    dest="output_zip",
                    type=str,
                    default="output.zip",
                    help="Zip archive to dump the created logs.")

args = parser.parse_args()
print(f"{bcolors.HEADER}Passed: {args}{bcolors.ENDC}")

file = open(args.phrases)
phrases = file.read().split("\n")

print(f"{bcolors.OKCYAN}")
for p in phrases:
    print(p)
print(f"{bcolors.ENDC}")

for i in range(0, args.overall_runs):
    for m in models:
        start = time.time()
        filename = f"{m.name()}_{i}_.csv"

        extract_command = f"/DeepRecon/attacks/flush_reload . {filename}"
        p0 = subprocess.Popen(extract_command, shell=True)
        print(f"{bcolors.WARNING}Started symbol discovery{bcolors.ENDC}")

        while not os.path.exists("DR_ACTIVE"):
            time.sleep(0.5)
        print(f"{bcolors.WARNING}Started extraction{bcolors.ENDC}")

        t = threading.Thread(target=m.run, name=m.name(), args=(phrases,))
        t.start()
        print(f"{bcolors.WARNING}Started inference{bcolors.ENDC}")

        done = False
        while not done:
            time.sleep(0.5)
            done = True
            if p0.poll() is not None and t.isAlive() is True:
                done = False

        print(f"{bcolors.OKGREEN}Finished, zipping...{bcolors.ENDC}")

        zip_command = f"zip -rum {args.output_zip} {filename}"
        p1 = subprocess.Popen(zip_command, shell=True)
        while p1.poll() is None:
            time.sleep(0.5)
        
        print(f"{bcolors.OKGREEN}Zipped! Total time: {time.time() - start}{bcolors.ENDC}")

print(f"{bcolors.HEADER}Complete!{bcolors.ENDC}")