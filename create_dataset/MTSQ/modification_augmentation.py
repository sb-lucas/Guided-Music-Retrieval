import random
random.seed(42)  # for reproducibility

#-------------------------------------AUGMENTATION PART-------------------------------------#
ADD_TEMPLATES = [
    "add {tag}",
    "something like this but with {tag}",
    "this track, but with {tag}",
    "similar to this, but add {tag}",
    "find a version of this with {tag}",
    "same song but with {tag}",
    "this, plus {tag}",
    "a track with the same vibe but featuring {tag}",
    "i'm looking for this, but with {tag}",
    "like this, but with a {tag} in it",
    "same feel, but looking for one with {tag}",
    "a version of this track that includes {tag}",
    "find me something similar that has {tag}",
    "this reference track, but add {tag}"
]

REMOVE_TEMPLATES = [
    "remove {tag}",
    "something like this but without {tag}",
    "this track, but no {tag}",
    "similar to this, but drop the {tag}",
    "find a version without the {tag}",
    "same song but minus the {tag}",
    "this, but without {tag}",
    "a track with the same vibe but no {tag}",
    "i'm looking for this, but without {tag}",
    "like this, but excluding the {tag}",
    "same feel, but looking for one without {tag}",
    "a version of this track that doesn't have {tag}",
    "find me something similar but without {tag}",
    "this reference track, but remove the {tag}"
]

MAKE_IT_TEMPLATES = [
    "make it {tag}",
    "a more {tag} version of this",
    "this track, but more {tag}",
    "similar to this, but {tag}",
    "find something like this that is more {tag}",
    "same song but more {tag}",
    "a {tag} version of this",
    "like this, just more {tag}",
    "i want a track like this that is more {tag}",
    "find a {tag} alternative to this",
    "same vibe but leans more {tag}"
]

MAKE_IT_NOT_TEMPLATES = [
    "make it not {tag}",
    "a less {tag} version of this",
    "this track, but less {tag}",
    "similar to this, but not as {tag}",
    "find something like this that is less {tag}",
    "same song but not so {tag}",
    "like this, just less {tag}",
    "i want a track like this that isn't so {tag}",
    "a version of this that is less {tag}",
    "find an alternative that is not as {tag}"
]

FASTER_TEMPLATES = [
    "make it faster",
    "not so slow",
    "i want it faster",
    "a faster version of this",
    "this track, but faster",
    "similar to this, but higher tempo",
    "find something like this but faster",
    "same song but faster",
    "like this, just faster",
    "a higher bpm version of this",
    "looking for a faster track like this"
]

SLOWER_TEMPLATES = [
    "make it slower",
    "not so fast",
    "i want it slower",
    "a slower version of this",
    "this track, but slower",
    "similar to this, but lower tempo",
    "find something like this but slower",
    "same song but slower",
    "like this, just slower",
    "a lower bpm version of this",
    "looking for a slower track like this"
]

def augment_modification(modification):
    #there are 6 types of modifications:
    # make it {tag} -> tempo/mood
    # make it not {tag} -> tempo/mood
    # add {tag}
    # remove {tag} -> instruments
    #make it slower
    #make it faster

    if modification == "make it faster":
        return random.choice(FASTER_TEMPLATES)

    if modification == "make it slower":
        return random.choice(SLOWER_TEMPLATES)

    if modification.startswith("make it not "):
        tag = modification.split("make it not ")[1]
        return random.choice(MAKE_IT_NOT_TEMPLATES).format(tag=tag)
    
    if modification.startswith("make it "):
        tag = modification.split("make it ")[1]
        return random.choice(MAKE_IT_TEMPLATES).format(tag=tag)
    
    if modification.startswith("add "):
        tag = modification.split("add ")[1]
        return random.choice(ADD_TEMPLATES).format(tag=tag)
    
    if modification.startswith("remove "):
        tag = modification.split("remove ")[1]
        return random.choice(REMOVE_TEMPLATES).format(tag=tag)