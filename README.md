# M4 Training Experiments Logbook and Knowledge Memos

This repo aims at keeping track of training logs and knowledge-sharing memos for the M4 project.

Besides the usual experiment entries in the [training repo](https://github.com/huggingface/m4/), this is a place for logs with reproduction steps and potentially multiple log files, core dumps, graphs, actions taken after incidents, etc. We use it to keep the m4 repo small and not bloated with transient data. While this is primarily for internal use, since there are multiple instructions on how debug and analysis of problems, these logs can be invaluable to some people.

We additinoally share high level knowledge memos that summarize every once in a while the core learnings.


## How to organize and what to log

Feel free to organize and name the sub-folders in any way you like, grouping experiments that try to resolve a single issue together and if there multiple sub-experiments giving a common README that helps the reader navigate the multitude of entries and/or showing the critical successful entries. e.g. for a possible approach see [tr_141-hanging](./tr_141-hanging). But please use whatever is most productive for you and you reading this 6-12 months later.

Try to log as much pertinent information as possible, including relevant snippets of log files - so that if 6-12 months later you need to come back to understand an old experiment all the essential bits of information ideally would be here.
