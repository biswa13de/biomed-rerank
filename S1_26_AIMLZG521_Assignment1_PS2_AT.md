**Problem Statement 2**

# Cross-Encoder Reranking for Biomedical Literature Retrieval

**Conversational AI (AIMLCZG521) — EC-1 Assessment**

**Total Marks: 10**

**<u>General Instructions</u>**

1.  Students must carefully read and follow the instructions provided in each problem statement and task.

2.  Students must submit the completed assignment notebook in both

    1.  Jupyter Notebook format (.ipynb)

    2.  PDF format of the executed notebook

3.  The Jupyter Notebook must contain the output displayed for every executed cell.

4.  Assignments developed or submitted using any Python IDE other than the prescribed environment will not be considered for grading.

5.  Students must ensure that they submit the correct Assignment Set assigned to them. Submission of an incorrect Assignment Set will not be evaluated.

6.  For each and every task, students must provide a detailed explanation, justification, and inference. Merely providing code and output will not be sufficient for evaluation.

7.  The notebook should clearly include:

    1.  Assignment title

    2.  Student details

    3.  Problem statement

    4.  Dataset details and source

    5.  Tools and libraries used

    6.  Code implementation

    7.  Output screenshots/results

    8.  Explanation of the logic used

    9.  Justification for the chosen method/model/approach

    10. Inference drawn from the results

    11. Limitations observed

    12. Possible improvements

    13. Final conclusion

    14. References

8.  Students must clearly explain the architectural design, workflow, and reasoning behind the selected approach. For RAG-based assignments, this includes chunking, retrieval strategy, ranking/re-ranking, context construction, and evaluation. For fine-tuning-based assignments, this includes dataset preparation, training strategy, model adaptation method, preference optimization, and validation.

9.  Students must not simply use built-in APIs or pre-trained pipelines without explaining the underlying process. The implementation should clearly demonstrate conceptual understanding of the selected method.

10. Incomplete submissions, missing outputs, unclear explanations, absence of either the .ipynb file or the final PDF report, missing Virtual Lab screenshots, or lack of proper justification and inference may lead to a reduction of marks.

11. Students must prepare a final report in PDF format and upload it along with the .ipynb notebook. If the final PDF report is not attached, the submission will be considered partial, and marks will be reduced accordingly.

12. Students are advised to verify the complete notebook execution from start to end before submission. Notebooks with execution errors, missing cells, broken links, or unavailable datasets may be penalized.

13. Any copied content, direct use of AI-generated answers without proper understanding, or submission without meaningful explanation and inference may lead to mark deduction.

14. Proper references must be provided for datasets, models, libraries, research papers, APIs, documentation, or external resources used in the assignment.

15. Students must ensure that all figures, tables, architecture diagrams, workflow diagrams, and evaluation results are properly labelled and explained.

16. The final conclusion must summarize the key observations, strengths of the implemented approach, limitations, and possible future improvements.

**Total Marks: 10**

**Problem Statement**

Develop a Biomedical Literature Retrieval and Reranking System that combines an initial retrieval method with a Transformer-based cross-encoder reranking stage.

The system should first retrieve potentially relevant biomedical research documents or abstracts using BM25 or dense vector retrieval or Hybrid retrieval. A pre-trained cross-encoder should then examine each query-document pair and assign a relevance score. The retrieved documents should be reordered based on these scores.

The system should demonstrate how modern information retrieval systems use a fast candidate retrieval stage followed by a more accurate reranking stage to improve the relevance of top-ranked biomedical research documents.

The assignment focuses on biomedical literature search rather than medical diagnosis or treatment recommendations.

**Dataset**

Students may use publicly available biomedical and scientific information-retrieval datasets such as SciFact, BioASQ, PubMed abstracts, CORD-19, BEIR biomedical datasets, or another suitable public research-paper retrieval dataset.

Students should clearly document the dataset source, number of queries, number of documents/abstracts, relevance-label format, and train/test split if available.

**Module 1: Data Preparation & Initial Retrieval**

**Task 1: Biomedical Data Cleaning and Preparation (1 Mark)**

Prepare a collection of biomedical documents/abstracts and natural-language research queries.

The preprocessing workflow may include removal of duplicate records, handling missing fields, removal of unwanted formatting or HTML, text normalization, and preparation of query-document relevance labels.

The processed dataset should be saved in a structured format that can be used for indexing and retrieval.

**<u>Analyze and report:</u>**

- Why preprocessing is important for biomedical literature retrieval

- Challenges associated with long scientific abstracts

- Challenges caused by medical terminology, abbreviations, and named entities

- Challenges caused by similar terminology appearing in unrelated research

**Task 2: Initial Biomedical Search System (2 Marks)**

Build an initial retrieval pipeline using BM25, TF-IDF, Elasticsearch/OpenSearch, or dense vector retrieval.

The system should retrieve Top-K relevant biomedical documents or abstracts for research-related queries and rank them based on retrieval scores.

Students should record the initial ranking and retrieval latency.

**<u>Provide a brief analysis discussing:</u>**

- Advantages and limitations of the selected retrieval method

- Keyword mismatch and semantic mismatch challenges

- Cases where the initial retrieval system retrieves relevant but lower-ranked documents

**Module 2: Transformer-Based Cross-Encoder Reranking**

**Task 3: Cross-Encoder Reranking (2 Marks)**

Use a suitable pre-trained Transformer cross-encoder to rerank the candidate biomedical documents.

For each query, create query-document pairs and obtain a relevance score from the cross-encoder. Sort the candidate documents using the predicted relevance scores and return the final Top-K results.

Students are not required to train the cross-encoder from scratch.

**<u>Provide a brief explanation discussing:</u>**

- The difference between bi-encoders and cross-encoders

- How a cross-encoder models query-document interaction

- Why cross-encoders are useful for reranking

- Why reranking is applied only to a smaller candidate set

- How biomedical terminology can affect relevance scoring

**Task 4: Before vs After Reranking (2 Marks)**

Compare the initial retrieval ranking with the cross-encoder reranked results.

**<u>Students must provide at least five biomedical research queries and demonstrate:</u>**

- A relevant document promoted by reranking

- An irrelevant document moved lower in the ranking

- A case where ranking changes very little

- A failure case where reranking does not improve relevance

- A case involving domain-specific terminology or abbreviations

For each example, explain why the ranking changed and whether the final ranking is more useful for the user.

**Module 3: Retrieval Evaluation & Performance Analysis**

**Task 5: Retrieval and Reranking Evaluation (2 Marks)**

Evaluate both the initial retrieval and reranked results using:

Precision@5  
Recall@5  
MRR (Mean Reciprocal Rank)  
NDCG@5

Students must evaluate the system using at least 10 biomedical research queries and compare the initial and reranked results.

**<u>Students should also discuss:</u>**

- Limitations of cross-encoder reranking

- Retrieval quality versus reranking latency

- Cases where reranking provides significant improvement

- Cases where reranking provides little or no improvement

**Task 6: Candidate Size Experiment (1 Mark)**

Conduct one controlled experiment by comparing different candidate-pool sizes, such as Top-5, Top-10, and Top-20.

Alternatively, students may compare two suitable pre-trained cross-encoder models.

Analyze the effect on ranking quality and inference latency.

Students should explain the trade-off between retrieving more candidates and the computational cost of reranking them.

**Module 4: Error Analysis**

**Task 7: Reranking Error Analysis (1 Mark)**

**<u>Identify at least five reranking failures and analyze possible causes such as:</u>**

- Medical terminology mismatch

- Abbreviation ambiguity

- Keyword overlap without true relevance

- Research-topic mismatch

- Negation or context misunderstanding

- Long-document truncation

- Insufficient context in abstracts

- Named-entity mismatch

Students should suggest possible improvements to the retrieval or reranking pipeline for the identified failure cases.

**Suggested Workflow**

Accept biomedical research query  
↓  
Execute BM25 / Dense Retrieval  
↓  
Retrieve Top-K candidate documents  
↓  
Create Query + Document pairs  
↓  
Apply pre-trained Cross-Encoder  
↓  
Generate relevance scores  
↓  
Rerank candidate documents  
↓  
Return final Top-K documents  
↓  
Evaluate initial vs reranked results

**Note**

Students should implement an end-to-end retrieval and reranking pipeline first and then conduct the required experiment.

A pre-trained cross-encoder must be used; students are not expected to train a Transformer reranker from scratch.

The assignment should focus on understanding candidate retrieval, query-document interaction, Transformer-based reranking, ranking evaluation, and relevance-versus-latency trade-offs.

The system is intended for biomedical literature retrieval and should not be used to provide medical diagnosis, treatment recommendations, or patient-specific medical advice.

## Deliverables

Students must submit:

1.  Python notebook (.ipynb) with results

2.  PDF of the notebook with results,

    1.  documentation

    2.  comparative tables and visualizations, 

    3.  Output examples with analysis

    4.  Recommendations based on their experiments as an actionable report

All files must be submitted as a **single .zip file**.

## Experimental Constraint

1.  The **model, dataset, hardware, prompts, and inference settings must remain consistent** across configurations to ensure a fair comparison.

2.  The result document should not be more than 30 pages.
