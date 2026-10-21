"""
Edge deployment benchmarking utilities.

Latency is measured for the complete processing interval from frame processing to metadata generation.
"""

import time


class DeploymentBenchmark:
    """Measure processing latency and throughput"""

    def __init__(self):
        self.latencies = []

    def measure(self, pipeline, frame):
        """Process one frame and record elapsed time"""

        start = time.perf_counter()
        result = pipeline.process_frame(frame)
        elapsed = time.perf_counter() - start
        self.latencies.append(elapsed)
        return result

    @property
    def average_latency_ms(self):
        """Average processing latency in milliseconds"""

        if not self.latencies:
            return 0.0

        return (sum(self.latencies)/ len(self.latencies) * 1000.0)

    @property
    def fps(self):
        """Average processing throughput"""

        if not self.latencies:
            return 0.0

        average_latency = (sum(self.latencies) / len(self.latencies))

        if average_latency == 0:
            return 0.0

        return 1.0 / average_latency