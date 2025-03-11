import concurrent.futures

def simple_task(a, b, c, d, e, f, g):
    return 0

if __name__ == "__main__":
    tasks = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(1000):
            for j in range(i + 1):
                tasks.append(executor.submit(simple_task, "a", "b", i, j, 1, 2, 3))

    for future in concurrent.futures.as_completed(tasks):
        future.result()
    print("finished")
