import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def comparePerformance(firstResults, secondResults, thirdResults):
  x = ['Phase']
  firstY = firstResults
  secondY = secondResults
  thirdY = thirdResults
  
  xAxis = np.arange(len(x)) 
  plt.bar(xAxis - 0.1, firstY, 0.1, label = 'Without PET') 
  plt.bar(xAxis, secondY, 0.1, label = 'With HE') 
  plt.bar(xAxis + 0.1, thirdY, 0.1, label = 'With MPC') 
    
  plt.xticks(xAxis, x) 
  plt.ylabel("Accuracy") 
  plt.title("Comparison of the performance in different phases:") 
  plt.legend() 
  plt.grid(True, linestyle = '--', linewidth = 0.5)
  plt.show() 


def main():
    accuracyBeforeFinetuning = 0.9649353391396148
    accuracyAfterFinetuning = 0.96456320929005
    accuracyAfterKnowledgeDistillation = 0.9178701504354713
    comparePerformance(accuracyBeforeFinetuning, accuracyAfterFinetuning, accuracyAfterKnowledgeDistillation)

    accuracyBeforeFinetuning = 0.45425101220607755
    accuracyAfterFinetuning = 0.5103913654883703
    accuracyAfterKnowledgeDistillation = 0.7112010717391968
    comparePerformance(accuracyBeforeFinetuning, accuracyAfterFinetuning, accuracyAfterKnowledgeDistillation)



if __name__ == "__main__":
    main()