class ChainOfThoughtState:
    """ We consider {context} with the {target} as utterance
        that experience emotion. This chain of thought help
        with reasoning on the potential sources (texts spans, i.e. utterances) of the
        emotion that causes and results in state of the {target}.
    """

    @staticmethod
    def prompt_for_span_inferring(context, target):
        new_context = f'Given the conversation "{context}", '
        prompt = new_context + f'which text spans are possibly causes emotion on {target} ?'
        return new_context, prompt

    @staticmethod
    def prompt_for_opinion_inferring(context, target, aspect_expr):
        new_context = context + ' The mentioned text spans are about ' + aspect_expr + '.'
        prompt = new_context + f' Based on the common sense, ' \
                               f'what is the implicit opinion towards the mentioned text spans that causes emotion on {target}, and why?'
        return new_context, prompt

    @staticmethod
    def prompt_for_emotion_inferring(context, target, opinion_expr):
        new_context = context + f' The opinion towards the text spans that causes emotion on {target} is ' + opinion_expr + '.'
        prompt = new_context + f' Based on such opinion, what is the emotion state of {target}?'
        return new_context, prompt

    @staticmethod
    def prompt_for_emotion_label(context, polarity_expr, label_list):
        prompt = context + f' The emotion state is {polarity_expr}.' + \
                 " Based on these contexts, summarize and return the emotion cause only." + \
                 " Choose from: {}.".format(", ".join(label_list))
        return prompt
