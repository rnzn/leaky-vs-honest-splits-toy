# In Plain English

I'm a wet-lab biochemist who built this through AI-assisted coding. This document is for anyone with bench experience who wants to expand into the in-silico world but finds GitHub intimidating. It walks through the code line by line in plain language.

This toy model sets up and answers one question: **if you test a model on proteins that are near-copies of its training proteins, does it look better than it really is?** The answer is yes, and this code measures by how much.

The usual way to train a model from a dataset is to split the data randomly into training and testing sets, train on one, and use the other to check if the model has learned anything useful. But for proteins, random splits are misleading — proteins violate the I.I.D. (Independent and Identically Distributed) assumption because they're often evolutionarily related (homologous). Imagine you have a hundred protein sequences and their melting temperatures. If you split randomly, members of the same family can end up on *both* sides, and the model can lean on family identity as a shortcut — inflating its apparent accuracy without actually learning what generalizes to a new family. That's a **leaky split**. An **honest split** keeps whole families on one side or the other, so the model is tested only on families it has never seen.

I'll walk through the code line by line in plain language.
