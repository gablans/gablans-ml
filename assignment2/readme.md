**This whole thing is highly likely to be completely useless.**

That is because jython is deprecated and unsupported by Python 3.X
It works well on Python 2.7, but the latest PIP update will deprecate everything 2.7 related.

My workaround was to make jython to work on 2.7 with the goal of generating output files based on the code from ABAGAIL. I then grabbed those files and created the Python 3.9 compatible 'plotting.py' file to be able to read and interpret the data so that I could create graphs.