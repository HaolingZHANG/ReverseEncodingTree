from tasks.task_inform import XOR, METHOD_TYPE, save_distribution

if __name__ == '__main__':
    xor_task = XOR(method_type=METHOD_TYPE.FS, max_generation=500, display_results=False)
    generations, counts = xor_task.run(2)
    save_distribution(counts=counts, parent_path="../output/", task_name="xor", method_type=METHOD_TYPE.FS)
