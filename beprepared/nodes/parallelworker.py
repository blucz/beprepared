import multiprocessing as mp
import time

# The base worker class.
class BaseWorker(mp.Process):
    def __init__(self, work_queue, result_queue, *args, **kwargs):
        """
        Accept extra parameters via *args and **kwargs.
        """
        super().__init__()
        self.work_queue = work_queue
        self.result_queue = result_queue
        # Store extra worker parameters for later use.
        self.worker_params = kwargs

    def initialize_worker(self):
        """
        Virtual method to run before processing starts.
        Override in your subclass.
        """
        pass

    def process_item(self, item):
        """
        Virtual method to process a single work item.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("process_item must be implemented by subclass.")

    def shutdown_worker(self):
        """
        Virtual method to run after all work is done.
        Override in your subclass if needed.
        """
        pass

    def run(self):
        # Run initialization code before processing.
        self.initialize_worker()

        while True:
            item = self.work_queue.get()
            # Use None as a sentinel to indicate no more work.
            if item is None:
                break
            try:
                result = self.process_item(item)
                # Send back a tuple (True, result) on success.
                self.result_queue.put((True, result))
            except Exception as e:
                # In case of an exception, send (False, exception).
                self.result_queue.put((False, e))
        self.shutdown_worker()


# The controller class.
class ParallelController:
    def __init__(self, worker_cls, worker_params_list):
        """
        :param worker_cls: A subclass of BaseWorker.
        :param worker_params_list: A list of parameter dictionaries, one for each worker.
        """
        self.worker_cls = worker_cls
        self.worker_params_list = worker_params_list  # List of dicts for each worker.
        self.work_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []

    def add_work_items(self, items):
        """
        Enqueue all work items.
        """
        for item in items:
            self.work_queue.put(item)

    def start(self):
        """
        Start a worker for each set of parameters in worker_params_list.
        """
        for params in self.worker_params_list:
            # Pass each set of worker parameters as keyword arguments.
            worker = self.worker_cls(self.work_queue, self.result_queue, **params)
            worker.start()
            self.workers.append(worker)

    def finish(self):
        """
        Enqueue a sentinel (None) for each worker to signal the end of work.
        """
        for _ in self.worker_params_list:
            self.work_queue.put(None)

    def run(self, items):
        """
        Enqueue work items, start the workers, and yield results as they come in.
        """
        self.add_work_items(items)
        self.start()
        self.finish()

        total_items = len(items)
        for _ in range(total_items):
            yield self.result_queue.get()

        # Wait for all workers to finish.
        for worker in self.workers:
            worker.join()


'''
# Example subclass of BaseWorker that uses the extra parameters.
class MyWorker(BaseWorker):
    def initialize_worker(self):
        # Access a parameter named 'param1' from the worker_params.
        param_value = self.worker_params.get('param1', 'default')
        print(f"Worker {self.pid}: starting up with param1 = {param_value}.")
        self.my_param = param_value

    def process_item(self, item):
        print(f"Worker {self.pid}: processing item {item} using param1 = {self.my_param}.")
        # Simulate some work.
        time.sleep(1)
        return item * 2

    def shutdown_worker(self):
        print(f"Worker {self.pid}: shutting down.")


# Example usage.
if __name__ == '__main__':
    # Build up a list of work items.
    items = list(range(10))
    
    # Provide a list of parameter dictionaries—one per worker.
    # For example, here we create three workers with different parameter values.
    worker_params_list = [
        {'param1': 'alpha'},
        {'param1': 'beta'},
        {'param1': 'gamma'},
    ]

    # Create a controller that will spawn one worker for each set of parameters.
    controller = ParallelController(MyWorker, worker_params_list)

    # Run the processing loop and handle results as they arrive.
    for success, result in controller.run(items):
        if success:
            print("Controller got result:", result)
        else:
            print("Controller got an error:", result)

'''
