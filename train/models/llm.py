from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

#https://huggingface.co/Qwen/Qwen3-30B-A3B#qwen3-highlights

#model_name = "Qwen/Qwen3-0.6B"
#model_name = "Qwen/Qwen3-4B"
#model_name = "Qwen/Qwen3-8B"
#model_name = "Qwen/Qwen3-14B"
#model_name = "Qwen/Qwen3-30B-A3B"
#model_name = "Qwen/Qwen3-32B"

class LLM_Based():
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/home/ldsbrandao/exp/hf_models")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/home/ldsbrandao/exp/hf_models"
        )

        print(f'llm device:{self.model.device}')

        self.system_prompt = """You are a helpful assistant that generates a new song description based on a previous song description 
        and a modification instruction. For example: Original song description: A rock song with electric guitar and drums that has 
        a happy mood. Modification: "i want a similar song, but that has a sad mood". New song description: A rock song with electric 
        guitar and drums that has a sad mood.""" 

        self.classifications = pd.read_csv("/home/ldsbrandao/exp/diego/diego/final/data/classifications/classifcations_for_llm.csv")

        self.tag_name_corrections = {
            'accordion': 'accordion',
            'acousticbassguitar': 'acoustic bass guitar',
            'acousticguitar': 'acoustic guitar',
            'bass': 'bass',
            'beat': 'beat',
            'bell': 'bell',
            'bongo': 'bongo',
            'brass': 'brass',
            'cello': 'cello',
            'clarinet': 'clarinet',
            'classicalguitar': 'classical guitar',
            'computer': 'computer',
            'doublebass': 'double bass',
            'drummachine': 'drum machine',
            'drums': 'drums',
            'electricguitar': 'electric guitar',
            'electricpiano': 'electric piano',
            'flute': 'flute',
            'guitar': 'guitar',
            'harmonica': 'harmonica',
            'harp': 'harp',
            'horn': 'horn',
            'keyboard': 'keyboard',
            'oboe': 'oboe',
            'orchestra': 'orchestra',
            'organ': 'organ',
            'pad': 'pad',
            'percussion': 'percussion',
            'piano': 'piano',
            'pipeorgan': 'pipe organ',
            'rhodes': 'rhodes',
            'sampler': 'sampler',
            'saxophone': 'saxophone',
            'strings': 'strings',
            'synthesizer': 'synthesizer',
            'trombone': 'trombone',
            'trumpet': 'trumpet',
            'viola': 'viola',
            'violin': 'violin',
            'voice': 'voice',

            'action': 'action',
            'adventure': 'adventure',
            'advertising': 'advertising',
            'background': 'background',
            'ballad': 'ballad',
            'calm': 'calm',
            'children': 'children',
            'christmas': 'christmas',
            'commercial': 'commercial',
            'cool': 'cool',
            'corporate': 'corporate',
            'dark': 'dark',
            'deep': 'deep',
            'documentary': 'documentary',
            'drama': 'drama',
            'dramatic': 'dramatic',
            'dream': 'dream',
            'emotional': 'emotional',
            'energetic': 'energetic',
            'epic': 'epic',
            'fast': 'fast',
            'film': 'film',
            'fun': 'fun',
            'funny': 'funny',
            'game': 'game',
            'groovy': 'groovy',
            'happy': 'happy',
            'heavy': 'heavy',
            'holiday': 'holiday',
            'hopeful': 'hopeful',
            'inspiring': 'inspiring',
            'love': 'love',
            'meditative': 'meditative',
            'melancholic': 'melancholic',
            'melodic': 'melodic',
            'motivational': 'motivational',
            'movie': 'movie',
            'nature': 'nature',
            'party': 'party',
            'positive': 'positive',
            'powerful': 'powerful',
            'relaxing': 'relaxing',
            'retro': 'retro',
            'romantic': 'romantic',
            'sad': 'sad',
            'sexy': 'sexy',
            'slow': 'slow',
            'soft': 'soft',
            'soundscape': 'soundscape',
            'space': 'space',
            'sport': 'sport',
            'summer': 'summer',
            'trailer': 'trailer',
            'travel': 'travel',
            'upbeat': 'upbeat', 
            'uplifting': 'uplifting',

            '60s': '60s style',
            '70s': '70s style',
            '80s': '80s style',
            '90s': '90s style',
            'acidjazz': 'acid jazz',
            'alternative': 'alternative',
            'alternativerock': 'alternative rock',
            'ambient': 'ambient',
            'atmospheric': 'atmospheric',
            'blues': 'blues',
            'bluesrock': 'blues rock',
            'bossanova': 'bossa nova',
            'breakbeat': 'breakbeat',
            'celtic': 'celtic',
            'chanson': 'chanson',
            'chillout': 'chillout',
            'choir': 'choir',
            'classical': 'classical',
            'classicrock': 'classic rock',
            'club': 'club',
            'contemporary': 'contemporary',
            'country': 'country',
            'dance': 'dance',
            'darkambient': 'dark ambient',
            'darkwave': 'dark wave',
            'deephouse': 'deep house',
            'disco': 'disco',
            'downtempo': 'downtempo',
            'drumnbass': 'drum n bass',
            'dub': 'dub',
            'dubstep': 'dubstep',
            'easylistening': 'easy listening',
            'edm': 'EDM',
            'electronic': 'electronic',
            'electronica': 'electronica',
            'electropop': 'electropop',
            'ethno': 'ethno',
            'eurodance': 'eurodance',
            'experimental': 'experimental',
            'folk': 'folk',
            'funk': 'funk',
            'fusion': 'fusion',
            'groove': 'groove',
            'grunge': 'grunge',
            'hard': 'hard',
            'hardrock': 'hard rock',
            'hiphop': 'hip-hop',
            'house': 'house',
            'idm': 'IDM',
            'improvisation': 'improvisation',
            'indie': 'indie',
            'industrial': 'industrial',
            'instrumentalpop': 'instrumental pop',
            'instrumentalrock': 'instrumental rock',
            'jazz': 'jazz',
            'jazzfusion': 'jazz fusion',
            'latin': 'latin',
            'lounge': 'lounge',
            'medieval': 'medieval',
            'metal': 'metal',
            'minimal': 'minimal',
            'newage': 'new age',
            'newwave': 'new wave',
            'orchestral': 'orchestral',
            'pop': 'pop',
            'popfolk': 'pop folk',
            'poprock': 'pop rock',
            'postrock': 'post rock',
            'progressive': 'progressive',
            'psychedelic': 'psychedelic',
            'punkrock': 'punk rock',
            'rap': 'rap',
            'reggae': 'reggae',
            'rnb': 'R&B',
            'rock': 'rock',
            'rocknroll': 'rock n roll',
            'singersongwriter': 'singer-songwriter',
            'soul': 'soul',
            'soundtrack': 'soundtrack',
            'swing': 'swing',
            'symphonic': 'symphonic',
            'synthpop': 'synthpop',
            'techno': 'techno',
            'trance': 'trance',
            'triphop': 'trip-hop',
            'world': 'world',
            'worldfusion': 'world fusion'
        }
    
    def get_tags(self, song_name):
        row = self.classifications[self.classifications['song'] == song_name].iloc[0]

        thresholds = {
            'instrument': 0.15,
            'genre': 0.2,
            'moodtheme': 0.1
        }

        top_k = {
            'instrument': 3,
            'genre': 2,
            'moodtheme': 2
        }

        tags_dict = {}
        

        for tag_type in ['instrument', 'genre', 'moodtheme']:
            tag_cols = [c for c in row.index if c.startswith(f"{tag_type}_")]
            tag_values = [(c, row[c]) for c in tag_cols]

            # sort all values once (needed for fallback)
            sorted_all = sorted(tag_values, key=lambda x: x[1], reverse=True)

            # filter by threshold
            filtered = [(c, v) for c, v in sorted_all if v > thresholds[tag_type]]

            if len(filtered) > 1:
                selected = filtered[:top_k[tag_type]]
            else:
                # fallback: always take the best available (ensures at least one)
                selected = sorted_all[:1]

            # remove prefix
            tags = [c.replace(f"{tag_type}_", "") for c, _ in selected]

            # apply corrections
            tags = [self.tag_name_corrections.get(tag, tag) for tag in tags]

            tags_dict[tag_type] = tags

        return tags_dict
    
    def get_song_description(self, song_name):
        tags_dict = self.get_tags(song_name)

        description = f"A {', '.join(tags_dict['genre'])} song with {', '.join(tags_dict['instrument'])} that has a {', '.join(tags_dict['moodtheme'])} mood."

        return description

    def get_target_description(self, song_name, modification_text):
        reference_description = self.get_song_description(song_name)

        user_prompt = f"""
            Original song description: {reference_description}.
            Modification: {modification_text}.

            Modify the original song description according to the modification instruction and
            Write only the new song description as a single sentence.
            Do not include explanations, prefixes, or extra text.
            New song description:
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)


        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=1.5,
            top_p=1,
            top_k=200,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        llm_answer = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return llm_answer