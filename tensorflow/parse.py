import argparse
parser = argparse.ArgumentParser()
#parser.parse_args()

parser.add_argument("--square", help="Display the number whose square u want to find", type= int,
					default=30)
parser.add_argument("--cube", help="The number whose cube u want to find out", type=str,
					default=str(80))
args = parser.parse_args()

print(args.square**2)
print(int(args.cube)*int(args.cube)*int(args.cube))