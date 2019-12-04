from tasks.task_inform import *

if __name__ == '__main__':
    task = Logic(method_type=METHOD_TYPE.GS, logic_type=LOGIC_TYPE.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", METHOD_TYPE.GS)
