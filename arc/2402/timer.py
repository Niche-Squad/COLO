import time


class Timer:
    def __init__(self):
        self.start = time.time()

    def report(self) -> float:
        time_elapsed = time.time() - self.start
        print(
            "Time elapsed: {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        return time_elapsed
