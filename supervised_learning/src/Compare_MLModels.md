<h2> Machine Language Classifier Comparison </h2>

<section>
  <h4>Models used:</h4>
  <ol>
    <li>Naïve Bayes (NBClassifier) </li>
    <li>KNN (KNeighborsClassifier) </li>
    <li>SVM (svm.LinearSVC) </li>
    <li>Decision Tree (DecisionTreeClassifier) </li>
    <li>Random Forest (RandomForestClassifier) </li>
  </ol>
</section>
<br/>
<section>
 <h4>Naive Bayes</h4>
 <img src="../build/compare-gaussian-naive-bayes.png"
  alt="Naive Bayes image" width="500"
  />
  <p>Figure 1</p>
  <p>In Figure 1, the mean ratio of the True False Positives and the True Negative is 48 out of 50 or 96%</p>
</section>

 <h4>KNN</h4>
 <img src="../build/compare-k-nearest-neighbor.png"
 alt="KNN image" width="500"
 />
  <p>Figure 2</p>
  <p>In Figure 2, the mean ratio of the True False Positives and the True Negative is 48.33 out of 50 or 96.67%</p>

 <h4>SVM or Linear SVC</h4>
 <img src="../build/compare-linear-svc.png"
 alt="SVM image" width="500"
 />
  <p>Figure 3</p>
  <p>In Figure 3, the mean ratio of the True False Positives and the True Negative is 48.33 out of 50 or 96.67%</p>

 <h4>Decision Tree</h4>
 <img src="../build/compare-decision-tree.png"
 alt="Naive Bayes" width="500"
 />
  <p>Figure 4</p>
  <p>In Figure 4, the mean ratio of the True False Positives and the True Negative is 47.67 out of 50 or 95.33%</p>

 <h4>Random Forest</h4>
 <img src="../build/compare-random-forest.png"
 alt="Naive Bayes" width="500"
 />
  <p>Figure 5</p>
  <p>In Figure 5, the mean ratio of the True False Positives and the True Negative is 48 out of 50 or 96%</p>

  <section>
    <h3>Conclusion</h3>
    <p>KNN (K-Nearest Neighbor) and Linear SVC are tied for the most accurate results. While the Decision Tree made the least accurate predictions in the group. All the members of the group made correct predictions for the Iris-Setosa 100% of the time. KNN produced the best prediction for Iris-Virginica </p>
  <section>
