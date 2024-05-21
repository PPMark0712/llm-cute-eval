import os
import glob
if __name__ == "__main__":
    l = glob.glob("/data1/yyz/projects/evaluate/dcy_eval/data/tasks/mmlu/dev/*.csv")
    l1 = [x.split("/")[-1][:-len("_dev.csv")] for x in l]
    l1.sort()
    print(l1)
    