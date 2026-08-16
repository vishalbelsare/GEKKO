.. _ml:

Machine Learning
================

.. py:module:: gekko.ML

The :mod:`gekko.ML` module converts selected, already-fitted machine
learning regressors into GEKKO expressions. A converted prediction can be
used as an objective, a constraint, an intermediate expression, or part of
a dynamic model.

GEKKO does not train these estimators and does not call arbitrary Python
prediction functions while a solver is evaluating a model. Instead, an ML
interface extracts the fitted parameters and rebuilds the prediction with
GEKKO operators. Train and validate the estimator with its native library
first, then create the GEKKO representation.

Typical applications include data-driven design optimization, hybrid
first-principles/data-driven modeling, model predictive control, and
uncertainty-aware optimization.

Installation
------------

GEKKO and scikit-learn are sufficient for the introductory examples:

.. code-block:: console

   python -m pip install gekko scikit-learn scipy

TensorFlow, GPflow, and linear-tree are optional and are needed only for
the interfaces that use those packages.

Quick start: optimize a Gaussian process
----------------------------------------

The following example trains a scikit-learn Gaussian process on noisy data,
converts the fitted model to a GEKKO expression, and minimizes the predicted
response over ``0 <= x <= 1``.

.. code-block:: python

   import numpy as np
   from gekko import GEKKO
   from gekko.ML import Gekko_GPR
   from sklearn.gaussian_process import GaussianProcessRegressor
   from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

   # Reproducible training data
   rng = np.random.default_rng(7)
   X = np.linspace(0.0, 1.0, 41).reshape(-1, 1)
   y = np.cos(2.0 * np.pi * X[:, 0]) + rng.normal(0.0, 0.08, X.shape[0])

   # Train and validate the model outside GEKKO
   kernel = ConstantKernel(1.0) * RBF(0.2) + WhiteKernel(0.01)
   gpr = GaussianProcessRegressor(
       kernel=kernel,
       alpha=1.0e-6,
       normalize_y=True,
       random_state=7,
   )
   gpr.fit(X, y)

   # Rebuild the fitted prediction as a GEKKO expression
   m = GEKKO(remote=False)
   x = m.Var(value=0.5, lb=0.0, ub=1.0)
   # Object arrays avoid a NumPy 2 copy=False issue in the current wrapper.
   x_features = np.asarray([x], dtype=object)
   gpr_gekko = Gekko_GPR(gpr, m)
   y_hat = gpr_gekko.predict(x_features)

   m.Minimize(y_hat)
   m.solve(disp=False)

   print(f"x = {x.value[0]:.6f}")
   print(f"predicted minimum = {y_hat.value[0]:.6f}")

The optimized result depends on the fitted model, not directly on the
unknown source function. Keep decision-variable bounds within the region
covered by the training data unless extrapolation has been validated.

Recommended workflow
--------------------

#. Prepare the data and establish a train/validation/test procedure.
#. Fit the estimator with scikit-learn, TensorFlow, GPflow, or linear-tree.
#. Check predictive accuracy and residual behavior before optimization.
#. Create a ``GEKKO`` model and bounded decision variables.
#. Construct the matching :mod:`gekko.ML` interface with the fitted estimator
   and the same GEKKO model.
#. Call ``predict`` with features in the exact order used during training.
#. Compare native-library and GEKKO predictions at several fixed points.
#. Use the GEKKO prediction in an objective, equation, or inequality.
#. Solve and confirm that the solution remains inside a validated domain.

.. warning::

   Optimization can exploit small surrogate-model errors. A model with good
   average test accuracy may still produce a poor optimizer near a boundary,
   in a sparse region, or outside the training domain. Validate the optimized
   point with the original model and, when possible, with the physical system
   or a higher-fidelity model.

Available interfaces
--------------------

.. list-table:: Fitted-model interfaces
   :header-rows: 1
   :widths: 25 28 47

   * - GEKKO interface
     - Source estimator
     - Main notes
   * - :class:`Gekko_GPR`
     - scikit-learn ``GaussianProcessRegressor`` or selected GPflow models
     - Supports a GEKKO prediction and an optional predictive standard
       deviation.
   * - :class:`Gekko_SVR`
     - scikit-learn ``SVR`` or ``NuSVR``
     - Supports ``rbf``, ``poly``, ``linear``, and ``sigmoid`` kernels.
   * - :class:`Gekko_NN_Sklearn`
     - scikit-learn ``MLPRegressor``
     - Returns one GEKKO expression per output. Scaling is not automatic.
   * - :class:`Gekko_NN_TF`
     - TensorFlow/Keras Dense networks
     - Intended for Dense-layer models; ``predict`` returns the first output.
   * - :class:`Gekko_LinearRegression`
     - scikit-learn ``LinearRegression`` or ``Ridge``
     - Feature engineering must be reproduced with GEKKO expressions.
   * - :class:`Gekko_DecisionTree`
     - scikit-learn ``DecisionTreeRegressor``
     - Represents branch logic with ``GEKKO.if2`` or ``GEKKO.if3``.
   * - :class:`Gekko_RandomForest`
     - scikit-learn ``RandomForestRegressor``
     - Averages converted decision trees; model size grows with the forest.
   * - :class:`Gekko_GradientBooster`
     - scikit-learn ``GradientBoostingRegressor``
     - Converts the initial estimate and each fitted regression tree.
   * - :class:`Gekko_LinearTree`
     - ``lineartree.LinearTreeRegressor``
     - Uses tree branch logic with a linear model in each leaf.

The module also includes :class:`Bootstrap`, :class:`Conformist`, and
:class:`Delta` wrappers for selected uncertainty calculations, plus
:class:`Gekko_Scaled_Model` for scikit-learn scalers.

API reference
-------------

Gaussian process regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: Gekko_GPR(model, m, modelType='sklearn', fixedKernel=True)

   Convert a fitted Gaussian process regression model to GEKKO expressions.

   :param model: A fitted scikit-learn ``GaussianProcessRegressor``. When
      ``modelType`` is not ``'sklearn'``, the object is treated as a GPflow
      model and is converted through a temporary scikit-learn model.
   :param m: The ``GEKKO`` model that owns all generated
      expressions.
   :param str modelType: Use ``'sklearn'`` for a scikit-learn model or
      ``'gpflow'`` for the supported GPflow conversion path.
   :param bool fixedKernel: During GPflow conversion, fix the converted kernel
      hyperparameters when ``True``. This argument does not affect a supplied
      scikit-learn model.

   Supported scikit-learn kernels include ``RBF``, ``Matern`` with
   ``nu`` equal to ``0.5``, ``1.5``, or ``2.5``, ``ConstantKernel``,
   ``WhiteKernel``, ``RationalQuadratic``, ``ExpSineSquared``, and
   ``DotProduct``. Sum, product, and exponentiation compositions are parsed
   recursively. Custom kernels are not supported unless a corresponding
   GEKKO implementation is added.

   The GPflow path reconstructs and fits a scikit-learn Gaussian process from
   the GPflow data and selected kernel parameters. Compare its predictions
   with the original GPflow model before optimization.

   .. py:method:: predict(xi, return_std=False)

      Build the prediction at ``xi``. Pass a one-dimensional NumPy object
      array, such as ``np.asarray([x1, x2], dtype=object)``, to make feature
      order explicit and to avoid the NumPy 2 ``copy=False`` compatibility
      issue described in :ref:`ml-numpy-copy-error`.

      When ``return_std=True``, return ``(prediction, standard_deviation)`` as
      GEKKO expressions. The calculation increases model size with the number
      of Gaussian-process training samples.

Support vector regression
~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: Gekko_SVR(model, m)

   Convert a fitted scikit-learn ``SVR`` or ``NuSVR`` model.

   :param model: A fitted ``sklearn.svm.SVR`` or ``sklearn.svm.NuSVR``
      estimator.
   :param m: The owning ``GEKKO`` model.

   Supported kernel names are ``'rbf'``, ``'poly'``, ``'linear'``, and
   ``'sigmoid'``. Callable and precomputed kernels are not converted.

   .. py:method:: predict(xi, return_std=False)

      Return the GEKKO prediction. Pass ``xi`` as a one-dimensional NumPy
      object array with the training feature order. With ``return_std=True``,
      the second result
      is the estimator's ``epsilon`` attribute. It is a constant margin, not
      a statistical standard deviation or calibrated prediction interval; for
      ``NuSVR`` this value is typically zero.

Scikit-learn neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: Gekko_NN_Sklearn(model, m)

   Convert a fitted scikit-learn ``MLPRegressor``.

   :param model: A fitted ``sklearn.neural_network.MLPRegressor``.
   :param m: The owning ``GEKKO`` model.

   The wrapper copies the fitted weight matrices and bias vectors and rebuilds
   each dense layer. It does not copy a preprocessing pipeline and does not
   scale inputs or outputs automatically.

   The current implementation correctly maps the scikit-learn ``identity``,
   ``tanh``, and ``relu`` activation names. Avoid ``activation='logistic'``
   until the wrapper maps scikit-learn's ``'logistic'`` name to its sigmoid
   expression. A ReLU network introduces piecewise logic and can be more
   difficult to optimize than a smooth ``tanh`` network.

   .. py:method:: predict(x)

      Return a list of GEKKO expressions, one per model output. For a
      single-output regressor, select the scalar expression with ``[0]``.

   .. py:method:: __call__(x)

      Equivalent to :meth:`predict`.

TensorFlow/Keras neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: Gekko_NN_TF(model, m)

   Convert the Dense layers of a fitted TensorFlow/Keras model.

   :param model: A fitted Keras model whose non-input, non-dropout layers have
      Dense-style weights, biases, and activation metadata.
   :param m: The owning ``GEKKO`` model.

   Input and Dropout layers are skipped. Convolutional, recurrent, attention,
   normalization, and arbitrary custom layers are not converted by this
   interface. Validate every converted output against ``model.predict``.

   .. py:method:: forward(x)

      Return a list containing all output-layer expressions.

   .. py:method:: predict(x, return_std=False)

      Return only the first output expression. The current implementation
      accepts ``return_std`` for API compatibility but does not calculate or
      return an uncertainty value.

Linear regression
~~~~~~~~~~~~~~~~~

.. py:class:: Gekko_LinearRegression(model, m)

   Convert a fitted scikit-learn linear or ridge regression model.

   :param model: A fitted estimator with ``coef_`` and ``intercept_``
      attributes, such as ``LinearRegression`` or ``Ridge``.
   :param m: The owning ``GEKKO`` model.

   The current implementation expects a two-dimensional target during fitting,
   for example ``model.fit(X, y.reshape(-1, 1))``. Polynomial and other derived
   features must be reconstructed explicitly with GEKKO operators before
   calling ``predict``.

   .. py:method:: predict(xi, return_std=False)

      Return the linear prediction. Pass ``xi`` as a one-dimensional NumPy
      object array with the engineered features in training order. With
      ``return_std=True``, the second result
      is currently the constant value zero; use :class:`Delta` or another
      calibrated method when an interval is required.

Scaling wrapper
~~~~~~~~~~~~~~~

.. py:class:: Gekko_Scaled_Model(gmodel, scaler_x=None, scaler_y=None)

   Wrap a converted GEKKO model with fitted scikit-learn input and/or output
   scalers.

   :param gmodel: An already-converted GEKKO ML interface.
   :param scaler_x: A fitted ``StandardScaler`` or ``MinMaxScaler`` for input
      features.
   :param scaler_y: A fitted ``StandardScaler`` or ``MinMaxScaler`` for output
      values.

   .. py:method:: predict(X, return_std=False)

      Scale the inputs, evaluate the converted model, and transform the result
      back to the original output units. When the wrapped model supports
      ``return_std``, the uncertainty is rescaled without applying an output
      offset.

.. note::

   This wrapper is experimental in the current implementation. Explicit
   scaling, as shown in :ref:`ml-neural-network-scaling`, is easier to inspect
   and should be preferred until native and GEKKO predictions have been
   compared for the complete preprocessing path.

Tree-based regressors
~~~~~~~~~~~~~~~~~~~~~

Tree interfaces translate each split into GEKKO conditional expressions.
They are most practical for shallow trees. Deep trees and large ensembles can
create many conditional expressions and can substantially increase solve time.

``ifo=2`` selects ``GEKKO.if2`` and ``ifo=3`` selects ``GEKKO.if3``. The
``eps`` value shifts branch tests slightly to reduce ambiguity at an exact
split threshold. Check predictions on both sides of every relevant split when
changing either option.

.. py:class:: Gekko_DecisionTree(model, m, ifo=2, eps=1e-3)

   Convert a fitted scikit-learn ``DecisionTreeRegressor`` with a scalar
   output.

   :param model: A fitted ``DecisionTreeRegressor``.
   :param m: The owning ``GEKKO`` model.
   :param int ifo: Conditional formulation selector, either ``2`` or ``3``.
   :param float eps: Offset applied at branch thresholds.

   .. py:method:: predict(input, return_proba=False, return_conds=False)

      Return the tree prediction. With ``return_conds=True``, also return the
      list of leaf-activation expressions. With ``return_proba=True``, also
      return the fraction of root training samples assigned to the active leaf.
      This value is a leaf support fraction, not a classification probability.
      If both optional flags are true, ``return_conds`` takes precedence.

.. py:class:: Gekko_RandomForest(model, m, ifo=2, eps=1e-3)

   Convert a fitted scikit-learn ``RandomForestRegressor`` by converting and
   averaging its component trees.

   .. py:method:: predict(input)

      Return the mean of the converted tree predictions.

.. py:class:: Gekko_GradientBooster(model, m, ifo=2, eps=1e-3)

   Convert a fitted scikit-learn ``GradientBoostingRegressor``. This interface
   does not apply to histogram-based gradient boosting estimators.

   .. py:method:: predict(input)

      Return the initial estimate plus the learning-rate-weighted tree
      predictions.

.. py:class:: Gekko_LinearTree(model, m, ifo=2, eps=0)

   Convert a fitted ``lineartree.LinearTreeRegressor``. Each leaf contributes
   a local linear model when its branch conditions are active.

   .. py:method:: predict(input, return_conds=False)

      Return the prediction. With ``return_conds=True``, also return the list
      of leaf-activation expressions.

.. note::

   The random-forest, gradient-booster, scaled-model, and linear-tree wrappers
   should be treated as experimental. Compare their GEKKO predictions with the
   original estimators over a representative grid before using them in an
   optimization problem.

.. _ml-uncertainty:

Uncertainty wrappers
~~~~~~~~~~~~~~~~~~~~

The meaning of the second value returned by ``return_std=True`` is
interface-specific. It is not always a statistical standard deviation.

.. list-table:: Meaning of ``return_std``
   :header-rows: 1
   :widths: 28 72

   * - Interface
     - Second returned value
   * - :class:`Gekko_GPR`
     - Gaussian-process predictive standard deviation expression.
   * - :class:`Gekko_SVR`
     - Constant SVR epsilon margin; not a calibrated standard deviation.
   * - :class:`Gekko_LinearRegression`
     - Zero placeholder in the current implementation.
   * - :class:`Bootstrap`
     - Sample standard deviation across converted model predictions.
   * - :class:`Conformist`
     - User-supplied constant uncertainty half-width.
   * - :class:`Delta`
     - Student-t-based confidence or prediction half-width.

.. py:class:: Bootstrap(models, m)

   Combine predictions from two or more already-converted GEKKO models.

   :param models: A sequence of converted interfaces that all belong to the
      same GEKKO model and implement ``predict(xi)``.
   :param m: The owning ``GEKKO`` model.

   .. py:method:: predict(xi, return_std=False)

      Return the ensemble mean. With ``return_std=True``, also return the sample
      standard deviation across model predictions. At least two models are
      required for the standard-deviation calculation.

.. py:class:: Conformist(model, m, u)

   Attach a constant uncertainty margin to an already-converted model.

   :param model: A converted GEKKO model interface, not the original
      scikit-learn estimator.
   :param m: The owning ``GEKKO`` model.
   :param float u: A calibrated uncertainty half-width in output units.

   This class does not fit or calibrate a conformal predictor. Compute ``u``
   externally with a calibration set or a conformal-prediction package, then
   pass the result to this wrapper.

   .. py:method:: predict(xi, return_std=False)

      Return the base prediction. With ``return_std=True``, also return the
      constant margin ``u``.

.. py:class:: Delta(model, m, X, s)

   Add a first-order, least-squares-style uncertainty calculation to an
   already-converted model.

   :param model: A converted GEKKO model interface.
   :param m: The owning ``GEKKO`` model.
   :param X: The numeric design matrix used in the interval calculation.
   :param float s: Residual scale, commonly an estimated root-mean-square error.

   .. py:method:: predict(xi, return_std=False, conf=0.9, PI=0)

      Return the base prediction. With ``return_std=True``, also return a
      Student-t-based half-width. ``PI=0`` omits the additional prediction-error
      term; ``PI=1`` includes it. The feature vector ``xi`` and design matrix
      ``X`` must use the same feature construction, including any intercept
      column supplied by the user.

API changes from older examples
-------------------------------

Older GEKKO notebooks and documentation may contain names and signatures that
no longer match :mod:`gekko.ML`.

* Use ``Gekko_NN_Sklearn(model, m)``, not
  ``Gekko_NN_SKlearn(model, minMaxArray, m)``.
* ``CustomMinMaxGekkoScaler`` is not part of the current module. Use a fitted
  scikit-learn scaler and reproduce the transformation explicitly, or use the
  experimental :class:`Gekko_Scaled_Model` wrapper.
* Use ``Bootstrap`` with two ``o`` characters, not ``Boootstrap``.
* Pass an already-converted model to ``Conformist(model, m, u)``. Do not pass a
  list containing a native estimator and a margin.
* The neural-network wrappers do not automatically return uncertainty from a
  custom TensorFlow loss.
* ``Gekko_DecisionTree.return_proba`` reports active-leaf support for a
  regressor; it is not a selected-class probability.

Examples
--------

.. _ml-neural-network-scaling:

Explicit scaling for ``MLPRegressor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neural-network training often benefits from scaled inputs and outputs. Keep the
preprocessing explicit so that the same formulas are visible in both training
and optimization.

.. code-block:: python

   import numpy as np
   from gekko import GEKKO
   from gekko.ML import Gekko_NN_Sklearn
   from sklearn.neural_network import MLPRegressor
   from sklearn.preprocessing import StandardScaler

   rng = np.random.default_rng(7)
   X = np.linspace(0.0, 1.0, 80).reshape(-1, 1)
   y = np.cos(2.0 * np.pi * X[:, 0]) + rng.normal(0.0, 0.05, X.shape[0])

   x_scaler = StandardScaler().fit(X)
   y_scaler = StandardScaler().fit(y.reshape(-1, 1))
   X_scaled = x_scaler.transform(X)
   y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()

   mlp = MLPRegressor(
       hidden_layer_sizes=(20, 20),
       activation="tanh",
       max_iter=5000,
       random_state=7,
   )
   mlp.fit(X_scaled, y_scaled)

   m = GEKKO(remote=False)
   x = m.Var(value=0.5, lb=0.0, ub=1.0)

   # Repeat the fitted StandardScaler transformations with GEKKO expressions.
   x_scaled_m = (x - x_scaler.mean_[0]) / x_scaler.scale_[0]
   y_scaled_m = Gekko_NN_Sklearn(mlp, m).predict([x_scaled_m])[0]
   y_hat = m.Intermediate(
       y_scaled_m * y_scaler.scale_[0] + y_scaler.mean_[0]
   )

   m.Minimize(y_hat)
   m.solve(disp=False)

Bootstrap ensemble
~~~~~~~~~~~~~~~~~~

Train the native estimators first. After the GEKKO model is created, convert
each estimator with the same ``m`` and pass those converted interfaces to
:class:`Bootstrap`.

.. code-block:: python

   import numpy as np
   from gekko import GEKKO
   from gekko.ML import Bootstrap, Gekko_SVR
   from sklearn.svm import SVR

   rng = np.random.default_rng(7)
   X = np.linspace(0.0, 1.0, 60).reshape(-1, 1)
   y = np.cos(2.0 * np.pi * X[:, 0]) + rng.normal(0.0, 0.08, X.shape[0])

   estimators = []
   for _ in range(10):
       sample = rng.integers(0, X.shape[0], X.shape[0])
       estimator = SVR(C=10.0, epsilon=0.05, kernel="rbf")
       estimator.fit(X[sample], y[sample])
       estimators.append(estimator)

   m = GEKKO(remote=False)
   x = m.Var(value=0.5, lb=0.0, ub=1.0)
   x_features = np.asarray([x], dtype=object)
   converted = [Gekko_SVR(estimator, m) for estimator in estimators]
   ensemble = Bootstrap(converted, m)
   mean, std = ensemble.predict(x_features, return_std=True)

   # Example conservative objective for minimization.
   m.Minimize(mean + 1.645 * std)
   m.solve(disp=False)

Constant conformal margin
~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`Conformist` wrapper stores a margin; calibration remains an
external step. The example below computes a split-conformal absolute-residual
quantile and attaches it to a converted Gaussian process.

.. code-block:: python

   import numpy as np
   from gekko import GEKKO
   from gekko.ML import Conformist, Gekko_GPR
   from sklearn.gaussian_process import GaussianProcessRegressor
   from sklearn.gaussian_process.kernels import RBF, WhiteKernel
   from sklearn.model_selection import train_test_split

   rng = np.random.default_rng(7)
   X = np.linspace(0.0, 1.0, 80).reshape(-1, 1)
   y = np.cos(2.0 * np.pi * X[:, 0]) + rng.normal(0.0, 0.08, X.shape[0])
   X_train, X_cal, y_train, y_cal = train_test_split(
       X, y, test_size=0.25, random_state=7
   )

   gpr = GaussianProcessRegressor(
       kernel=RBF(0.2) + WhiteKernel(0.01),
       normalize_y=True,
       random_state=7,
   )
   gpr.fit(X_train, y_train)

   residual = np.abs(y_cal - gpr.predict(X_cal))
   alpha = 0.10
   level = min(1.0, np.ceil((len(residual) + 1) * (1 - alpha)) / len(residual))
   margin = float(np.quantile(residual, level, method="higher"))

   m = GEKKO(remote=False)
   x = m.Var(value=0.5, lb=0.0, ub=1.0)
   x_features = np.asarray([x], dtype=object)
   base = Gekko_GPR(gpr, m)
   calibrated = Conformist(base, m, margin)
   mean, half_width = calibrated.predict(x_features, return_std=True)

   # The constant margin affects a robust constraint, although it would not
   # change the minimizer if it were merely added to an objective.
   m.Equation(mean + half_width <= -0.5)
   m.Minimize(x)
   m.solve(disp=False)

Tree-model validation
~~~~~~~~~~~~~~~~~~~~~

A tree surrogate should be checked especially near split thresholds.

.. code-block:: python

   import numpy as np
   from gekko import GEKKO
   from gekko.ML import Gekko_DecisionTree
   from sklearn.tree import DecisionTreeRegressor

   X = np.linspace(0.0, 1.0, 80).reshape(-1, 1)
   y = np.cos(2.0 * np.pi * X[:, 0])
   tree = DecisionTreeRegressor(max_depth=4, random_state=7).fit(X, y)

   m = GEKKO(remote=False)
   x = m.Var(value=0.5, lb=0.0, ub=1.0)
   x_features = np.asarray([x], dtype=object)
   tree_gekko = Gekko_DecisionTree(tree, m, ifo=2, eps=1.0e-4)
   y_hat, leaf_conditions = tree_gekko.predict(
       x_features, return_conds=True
   )

   m.Minimize(y_hat)
   m.solve(disp=False)

   # At a fixed test point, compare tree.predict([[x_test]]) with the
   # corresponding GEKKO prediction before relying on the optimized result.

Dynamic optimization with a learned model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A converted model can appear in a differential equation. This example learns
``f(x1) = 0.5*x1**2`` and uses the learned function in a small dynamic
optimization problem.

.. code-block:: python

   import numpy as np
   from gekko import GEKKO
   from gekko.ML import Gekko_GPR
   from sklearn.gaussian_process import GaussianProcessRegressor
   from sklearn.gaussian_process.kernels import RBF, WhiteKernel

   # Train the static surrogate.
   X = np.linspace(-2.0, 2.0, 31).reshape(-1, 1)
   y = 0.5 * X[:, 0] ** 2
   gpr = GaussianProcessRegressor(
       kernel=RBF(0.7) + WhiteKernel(1.0e-8),
       normalize_y=True,
       random_state=7,
   ).fit(X, y)

   # Dynamic optimization model.
   m = GEKKO(remote=False)
   nt = 81
   m.time = np.linspace(0.0, 2.0, nt)

   x1 = m.Var(value=1.0)
   x2 = m.Var(value=0.0)
   u = m.MV(value=0.0, lb=-1.0, ub=1.0)
   u.STATUS = 1

   final_marker = np.zeros(nt)
   final_marker[-1] = 1.0
   final = m.Param(value=final_marker)

   state_features = np.asarray([x1], dtype=object)
   learned_rate = Gekko_GPR(gpr, m).predict(state_features)
   m.Equation(x1.dt() == u)
   m.Equation(x2.dt() == learned_rate)
   m.Minimize(final * x2)

   m.options.IMODE = 6
   m.solve(disp=False)

Troubleshooting
---------------

Prediction does not match the source estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check feature order, units, scaling, and output shape first. Evaluate both
models at fixed numeric points before adding an objective. For neural networks,
confirm that every activation is supported and remember that
:class:`Gekko_NN_Sklearn` returns a list.

.. _ml-numpy-copy-error:

``ValueError: Unable to avoid copy while creating an array``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current ``Gekko_GPR``, ``Gekko_SVR``, and
``Gekko_LinearRegression`` implementations call ``np.array(...,
copy=False)``. NumPy 2 raises ``ValueError`` when an input list or scalar
would require a copy. Until the implementation is changed to
``np.atleast_1d(np.asarray(xi, dtype=object))``, pass an existing object array
to ``predict``:

.. code-block:: python

   features = np.asarray([x1, x2], dtype=object)
   prediction = converted_model.predict(features)

``NameError: name 'm' is not defined``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some experimental wrappers in the current source use a module-level ``m`` in
parts of ``predict`` instead of the model stored by the instance. The source
implementation should use ``self.m`` consistently in
:class:`Gekko_Scaled_Model`, :class:`Gekko_RandomForest`,
:class:`Gekko_GradientBooster`, and :class:`Gekko_LinearTree`.

Tree model is slow or difficult to solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduce tree depth, prune the number of estimators, and validate a single
:class:`Gekko_DecisionTree` before converting an ensemble. Compare ``ifo=2``
and ``ifo=3`` formulations and tune ``eps`` only after checking predictions
near split thresholds.

Gaussian process creates a large GEKKO model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each prediction contains terms associated with the Gaussian-process training
samples. Reduce or summarize the training set, use a sparse modeling strategy,
or select a different surrogate when solve time or generated model size becomes
excessive.

Unexpected uncertainty result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Review :ref:`ml-uncertainty` and verify the units. The API name
``return_std`` is shared across interfaces, but the second value may be an
epsilon margin, a fixed half-width, or another interval measure.

Related packages
----------------

The ML interfaces build on models and utilities from the following projects:

* `scikit-learn <https://scikit-learn.org/stable/>`_
* `TensorFlow <https://www.tensorflow.org/>`_
* `GPflow <https://www.gpflow.org/>`_
* `linear-tree <https://github.com/cerlymarco/linear-tree>`_
* `SciPy <https://scipy.org/>`_

A conformal-prediction library may be used to calculate a calibration margin,
but :class:`Conformist` only stores and returns the supplied margin; it does
not depend on a particular calibration package.

Acknowledgements
----------------

The GEKKO machine-learning package was developed by graduate research
assistants `LaGrande Gunnell
<https://www.linkedin.com/in/lagrande-gunnell-715a2b194/>`_ and `Kyle
Manwaring <https://www.linkedin.com/in/kyle-manwaring-1310a1177/>`_. Thanks
to `John Vienna <https://www.linkedin.com/in/john-vienna-66b7a3219/>`_ and
`Xiaonan Lu <https://www.linkedin.com/in/xiaonan-lu-55775280/>`_ of Pacific
Northwest National Laboratory for technical direction and sponsorship through
a U.S. Department of Energy grant.
