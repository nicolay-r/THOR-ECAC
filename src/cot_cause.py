class ChainOfThoughtCause:
    """ We consider {context} with the {source} as utterance
        that (potentially) causes emotion to the predefined target,
        which is the terminate utterance of the conversation.
    """

    @staticmethod
    def prompt_for_span_inferring(context, source):
        new_context = f'Given the conversation "{context}", '
        prompt = new_context + f'which specific text span of {source} is possibly causes emotion?'
        return new_context, prompt

    @staticmethod
    def prompt_for_opinion_inferring(context, source, aspect_expr):
        new_context = context + ' The mentioned text span is about ' + aspect_expr + '.'
        prompt = new_context + f' Based on the common sense, ' \
                               f'what is the implicit opinion towards the cause of mentioned text span of {source}, and why?'
        return new_context, prompt

    @staticmethod
    def prompt_for_emotion_cause_inferring(context, source, opinion_expr):
        new_context = context + f' The opinion towards the text span of {source} that causes emotion is ' + opinion_expr + '.'
        prompt = new_context + f' Based on such opinion, what is the emotion caused by {source} towards the last conversation utterance?'
        return new_context, prompt

    @staticmethod
    def prompt_for_emotion_state_inferring(context, source, opinion_expr):
        """Note: utilized for Reasoning Revision of the emotion cause"""
        new_context = context + f' The opinion towards the text span of {source} that causes emotion is ' + opinion_expr + '.'
        prompt = new_context + f' Based on such opinion, what is the emotion state of {source}?'
        return new_context, prompt

    @staticmethod
    def prompt_for_emotion_cause_label(context, polarity_expr, label_list):
        prompt = context + f' The emotion caused is {polarity_expr}.' + \
                 " Based on these contexts, summarize and return the emotion cause only." + \
                 " Choose from: {}.".format(", ".join(label_list))
        return prompt

    @staticmethod
    def prompt_for_emotion_state_label(context, polarity_expr, label_list):
        """Note: utilized for Reasoning Revision of the emotion cause"""
        prompt = context + f' The emotion state is {polarity_expr}.' + \
                 " Based on these contexts, summarize and return the emotion state only." + \
                 " Choose from: {}.".format(", ".join(label_list))
        return prompt
