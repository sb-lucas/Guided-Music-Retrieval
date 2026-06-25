import random
random.seed(42)  # for reproducibility
#------------------------GETTING THE MODIFICATION BASED ON THE TAG FOR THE REAL DATASET------------------------#

binary_tags = ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad', 'acoustic', 'electronic']

other_tags = ['female', 'male', 'voice', 'instrumental']


#dictionary where the key is the reference song tag. Therefore, it assumes the target song has the opposite tag.
#Then, the value is a list of possible modification texts that can be used to create the triplet, that goes from
#reference song to target song. For example, if the reference song is instrumental and the target song is vocal, 
# the modification text can be "add vocals" or "make it vocal".
modifications = {}

for tag in binary_tags:
    modifications[tag] = f'make it not {tag}'
    modifications[f'not {tag}'] = f'make it {tag}'

modifications.update({
    'instrumental': 'add singing',
    'voice': 'remove singing',
    'female': 'make it male singing',
    'male': 'make it female singing',
    'bright': 'make it dark',
    'dark': 'make it bright',     
})

# this uses mod_text as key, and target tag as value. 
modifications_reverse = {}
for tag in binary_tags:
    modifications_reverse[f'make it not {tag}'] = f'not {tag}'
    modifications_reverse[f'make it {tag}'] = tag

#for the remaining
modifications_reverse.update({
    'add singing': 'voice',
    'remove singing': 'instrumental',
    'make it male singing': 'male',
    'make it female singing': 'female',
})

def get_modification_from_tag(tag):
    if tag in modifications:
        return modifications[tag]
    else:
        return None
    
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

def augment_modification(modification):
    #there are 4 types of modifications in the real dataset:
    # make it {tag} -> velocity/mood
    # make it not {tag} -> velocity/mood
    # add {tag}
    # remove {tag} -> instruments

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