from nlgeval import compute_metrics

metrics_dict = compute_metrics(hypothesis='Trans_ShellCodeGen_adj.csv',
                               references=['ShellCodeGen_references.csv'], no_skipthoughts=True,
                               no_glove=True)