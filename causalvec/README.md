Code for the paper "Distributed Representation of Words in Cause and Effect Spaces".


#### Experiment on English corpus

**Dataset**

The full train and test dataset can be found [here](http://clulab.cs.arizona.edu/data/emnlp2016-causal/).

**Train**

To train Max-Matching model, run

```bash
cd causalvec/en/
python maxmatching.py
```

This will generate 2 output files, as `cause_embedding.txt` and `effect_embedding.txt`.

**Test**

Using the generated causal embedding, the precision-recall doing of test word pairs can be calculate by the [code](https://github.com/clulab/releases/blob/master/emnlp2016-causal/src/main/scala/edu/arizona/sista/embeddings/DirectEval.scala). This will generate output file, in which each line is as `(recall, precision)`.

**Draw Precision-Recall Curve**

The generated file by previous step is used to draw Precision-Recall Curve of causal embedding model. Run

```bash
python draw_PR_curve.py
```

**Result**

![avatar](../fig/mm_PRCurve_en.png)

