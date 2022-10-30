import json
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from haystack.schema import Answer
from haystack.nodes.evaluator.evaluator import semantic_answer_similarity
from rest_api.schema import QuestionAnswerPair, PipelineHyperParams
from rest_api.utils import get_pipelines
from haystack import Pipeline
from document_indexing.s3_storage import S3Storage
from experiments.wandb_logger import WandBLogger


class PipelineEvaluation:
    @staticmethod
    def ranked_answers(answers: List[Answer]) -> List[str]:
        return [a.answer for a in sorted(answers, key=lambda a: a.score, reverse=True) if a.context]

    @staticmethod
    def get_n_worst_best_examples(
        questions: List[str],
        true_answers: List[str],
        scores: List[float],
        retrieved_questions: List[List[str]],
        n: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sorted_scores_index = np.argsort(scores)
        index_worst = sorted_scores_index[:n]
        index_best = sorted_scores_index[-n:]
        k = len(retrieved_questions[0])
        rank_cols = [f"rank{i}" for i in range(1, k + 1)]

        worst = pd.DataFrame(data=[retrieved_questions[i] for i in index_worst], columns=rank_cols)
        worst["question"] = [questions[i] for i in index_worst]
        worst["true_answer"] = [true_answers[i] for i in index_worst]

        best = pd.DataFrame(data=[retrieved_questions[i] for i in index_best], columns=rank_cols)
        best["question"] = [questions[i] for i in index_best]
        best["true_answer"] = [true_answers[i] for i in index_best]

        return worst[["question", "true_answer"] + rank_cols], best[["question", "true_answer"] + rank_cols]

    def get_performance_metrics(
        self, question_answer_pairs: List[QuestionAnswerPair], pipeline_hyper_params: PipelineHyperParams
    ) -> Tuple[Dict[str, float], Optional[Dict[str, pd.DataFrame]], Dict[str, Any]]:
        query_pipeline: Pipeline = get_pipelines(pipeline_hyper_params).get("query_pipeline", None)
        true_questions = []
        true_answers = []
        faq_answers = []
        extractive_answers = []

        # for each question answering pair, query the pipeline
        for q_and_a_pair in question_answer_pairs:
            true_answers.append([q_and_a_pair.answer])
            # use the alternative question for evaluation
            true_questions.append(q_and_a_pair.alternative_question)

            # faq pipeline retrieved answers
            result_faq = query_pipeline.run(
                query=q_and_a_pair.alternative_question, params={"CustomClassifier": {"index": "faq"}}
            )
            faq_answers.append(self.ranked_answers(result_faq["answers"]))

            # extractive pipeline extracted answers
            result_extractive = query_pipeline.run(
                query=q_and_a_pair.question, params={"CustomClassifier": {"index": "extractive"}}
            )
            extractive_answers.append(self.ranked_answers(result_extractive["answers"]))

        # for the faq, you can compute accuracy, reciprocal rank and sas
        rec_rank = self.reciprocal_rank(true_answers, faq_answers)
        top1_acc = self.top_k_accuracy(true_answers, faq_answers, k=1)
        topk_acc = self.top_k_accuracy(true_answers, faq_answers, k=pipeline_hyper_params.top_k)
        # also get some example of good and bad cases
        n_worst_reciprocal_rank, n_best_reciprocal_rank = self.get_n_worst_best_examples(
            true_questions, [ta[0] for ta in true_answers], rec_rank, faq_answers
        )
        faq_top1_sas, faq_topk_sas = self.mean_semantic_answer_similarity(true_answers, faq_answers)

        # for extractive, we can only do semantic similarity
        extr_top1_sas, extr_topk_sas = self.mean_semantic_answer_similarity(true_answers, extractive_answers)

        # also get some example of good and bad cases
        n_worst_sas, n_best_sas = self.get_n_worst_best_examples(
            true_questions, [ta[0] for ta in true_answers], extr_topk_sas, extractive_answers
        )

        examples = {
            "faq_worst_reciprocal_rank": n_worst_reciprocal_rank,
            "faq_best_reciprocal_rank": n_best_reciprocal_rank,
            "extractive_worst_sas_top_k": n_worst_sas,
            "extractive_best_sas_top_k": n_best_sas,
        }

        metrics = {
            "faq_mean_accuracy_top_1": sum(top1_acc) / len(top1_acc),
            "faq_mean_accuracy_top_k": sum(topk_acc) / len(topk_acc),
            "faq_mean_reciprocal_rank": sum(rec_rank) / len(rec_rank),
            "faq_mean_semantic_answer_similarity_top_1": sum(faq_top1_sas) / len(faq_top1_sas),
            "faq_mean_semantic_answer_similarity_top_k": sum(faq_topk_sas) / len(faq_topk_sas),
            "extractive_mean_semantic_answer_similarity_top_1": sum(extr_top1_sas) / len(extr_top1_sas),
            "extractive_mean_semantic_answer_similarity_top_k": sum(extr_topk_sas) / len(extr_topk_sas),
        }

        # plot relation between sas and ranking performance
        fig1 = plt.figure(f"top-{pipeline_hyper_params.top_k} semantic answer similarity versus reciprocal rank")
        plt.scatter(faq_topk_sas, rec_rank)
        plt.ylabel("reciprocal rank")
        plt.xlabel("mean semantic answer similarity")

        fig2 = plt.figure("top-1 semantic answer similarity versus accuracy")
        sns.boxplot(x=[int(i) for i in top1_acc], y=faq_top1_sas)
        plt.xlabel("accuracy")
        plt.ylabel("semantic answer similarity")

        return (
            metrics,
            examples,
            {
                f"top-{pipeline_hyper_params.top_k}-semantic-answer-similarity-versus-reciprocal-rank": fig1,
                "top-1-semantic-answer-similarity-versus-accuracy": fig2,
            },
        )

    @staticmethod
    def top_k_accuracy(true_answers: List[List[str]], retrieved_answers: List[List[str]], k) -> List[float]:
        accuracy = []
        for true_answer_list, retrieved_answer_list in zip(true_answers, retrieved_answers):
            retrieved_answer_list = retrieved_answer_list[:k]
            scores = [
                int(answer in true_answer) for answer in retrieved_answer_list for true_answer in true_answer_list
            ]
            accuracy.append(max(scores))
        return accuracy

    @staticmethod
    def reciprocal_rank(true_answers: List[List[str]], retrieved_answers: List[List[str]]) -> List[float]:
        """
        reciprocal rank = 1/position of first relevant answer
        """

        reciprocal_ranks = []

        for true_answer_list, retrieved_answer_list in zip(true_answers, retrieved_answers):
            scores = [
                int(answer in true_answer) / (1 + rank)
                for (rank, answer) in enumerate(retrieved_answer_list)
                for true_answer in true_answer_list
            ]
            reciprocal_ranks.append(max(scores))

        return reciprocal_ranks

    @staticmethod
    def mean_semantic_answer_similarity(
        true_answers: List[List[str]], retrieved_answers: List[List[str]]
    ) -> (float, float):
        top_1_sas, top_k_sas, _ = semantic_answer_similarity(
            predictions=retrieved_answers,
            gold_labels=true_answers,
            sas_model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            batch_size=32,
            use_gpu=False,
            use_auth_token=False,  # only needed of downloading private models from huggingface
        )
        return top_1_sas, top_k_sas


if __name__ == "__main__":

    # load Q and A pairs to evaluate on
    pipeline_hyper_params = PipelineHyperParams(**json.load(open("configuration.json", "r")))
    evaluator = PipelineEvaluation()
    storage = S3Storage()
    q_and_a_pairs = storage.load_qa_pairs("monopoly")
    metrics, example_tables, figures = evaluator.get_performance_metrics(q_and_a_pairs, pipeline_hyper_params)

    # log results to wandb
    logger = WandBLogger(project_name="monopoly", job_name="evaluate")
    for title, table in example_tables.items():
        logger.log_table(table, title)

    for figure_name, figure in figures.items():
        logger.log_figure(figure, figure_name, figure_name)

    logger.log_metrics(metrics)
    logger.log_metrics(pipeline_hyper_params.dict())
    logger.commit_logs()
