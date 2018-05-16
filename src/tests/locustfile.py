import json
from locust import HttpLocust, TaskSet, task


class ServiceTasks(TaskSet):
    def on_start(self):
        pass

    # @task
    # def index(self):
    #     self.client.get("/healthcheck")

    @task
    def siblya(self):
        self.client.post("/content-filtering/siblya", json.dumps({
            "data": [
                {
                    "cons": "fuck this shit, I can't lie .. I hate it here.",
                    "id": "1234123",
                    "kfcPred": 0,
                    "pros": "Overall so good, I enjoy working here",
                    "title": "A very good company",
                    "workLocation": "Australia"
                },
                {
                    "cons": "No comment here.",
                    "id": "1234124",
                    "kfcPred": 1,
                    "pros": "Great benefits",
                    "title": "What a abusive company",
                    "workLocation": "Australia"
                },
                {
                    "cons": "fuck this shit, I can't lie .. I hate it here.",
                    "id": "1234126",
                    "kfcPred": 0,
                    "pros": "Overall so good, I enjoy working here",
                    "title": "A very good company",
                    "workLocation": "Australia"
                },
                {
                    "cons": "fuck this shit, I can't lie .. I hate it here.",
                    "id": "1234125",
                    "kfcPred": 0,
                    "pros": "Overall so good, I enjoy working here",
                    "title": "A very good company",
                    "workLocation": "Australia"
                }
            ]
        }), headers={'Content-Type': 'application/json'})

    @task
    def kfc(self):
        self.client.post("/content-filtering/kfc", json.dumps({
            "reviews": [
                {
                    "id": "1234123",
                    "title": "A very good company",
                    "pros": "Overall so good, I enjoy working here",
                    "cons": "fuck this shit, I can't lie .. I hate it here.",
                    "workLocation": "Australia"
                },
                {
                    "id": "1234124",
                    "title": "What a abusive company",
                    "pros": "Great benefits",
                    "cons": "No comment here.",
                    "workLocation": "Australia"
                },
                {
                    "id": "1234126",
                    "title": "A very good company",
                    "pros": "Overall so good, I enjoy working here",
                    "cons": "fuck this shit, I can't lie .. I hate it here.",
                    "workLocation": "Australia"
                },
                {
                    "id": "1234125",
                    "title": "A very good company",
                    "pros": "Overall so good, I enjoy working here",
                    "cons": "fuck this shit, I can't lie .. I hate it here.",
                    "workLocation": "Australia"
                }
            ]
        }))


class ServiceUser(HttpLocust):
    task_set = ServiceTasks
    min_wait = 1000
    max_wait = 1000
