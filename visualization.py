import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

kevinJumpBP = pd.read_csv("./kevin/0.csv")
kevinJumpFP = pd.read_csv("./kevin/1.csv")
kevinJumpJP = pd.read_csv("./kevin/2.csv")
kevinJumpH = pd.read_csv("./kevin/3.csv")

kevinWalkBP = pd.read_csv("./kevin/4.csv")
kevinWalkFP = pd.read_csv("./kevin/5.csv")
kevinWalkJP = pd.read_csv("./kevin/6.csv")
kevinWalkH = pd.read_csv("./kevin/7.csv")

jacobJumpBP = pd.read_csv("./jacob/0.csv")
jacobJumpFP = pd.read_csv("./jacob/1.csv")
jacobJumpJP = pd.read_csv("./jacob/2.csv")
jacobJumpH = pd.read_csv("./jacob/3.csv")

jacobWalkBP = pd.read_csv("./jacob/4.csv")
jacobWalkFP = pd.read_csv("./jacob/5.csv")
jacobWalkJP = pd.read_csv("./jacob/6.csv")
jacobWalkH = pd.read_csv("./jacob/7.csv")

taylorJumpBP = pd.read_csv("./taylor/0.csv")
taylorJumpFP = pd.read_csv("./taylor/1.csv")
taylorJumpJP = pd.read_csv("./taylor/2.csv")
taylorJumpH = pd.read_csv("./taylor/3.csv")

taylorWalkBP = pd.read_csv("./taylor/4.csv")
taylorWalkFP = pd.read_csv("./taylor/5.csv")
taylorWalkJP = pd.read_csv("./taylor/6.csv")
taylorWalkH = pd.read_csv("./taylor/7.csv")
