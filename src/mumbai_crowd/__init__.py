"""Crowd-level prediction for the Mumbai Suburban Railway Harbour Line.

Capstone package.  Two course outcomes are implemented end to end:

* **CO2 - supervised regression.**  Predict coach-level crowd density from
  time, calendar, weather and operating features, trained under an
  *asymmetric* loss so that under-estimating danger is penalised far more
  heavily than over-estimating it.  See :mod:`mumbai_crowd.losses`,
  :mod:`mumbai_crowd.regression` and :mod:`mumbai_crowd.decision`.
* **CO5 - unsupervised clustering.**  Group the 35 Harbour-line stations by
  their 24-hour boarding/alighting signatures to recover the latent roles
  (dormitory, employment sink, interchange churn, dock belt).  See
  :mod:`mumbai_crowd.clustering`.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
