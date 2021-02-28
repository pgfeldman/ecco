from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from ecco.lm import LM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict

class EccoTrendAnalytics:
    """
    Class that uses the ecco library (https://github.com/jalammar/ecco) to record trends in language model behavior

    Attributes:
        lm:LM
            Ecco's central class. A wrapper around language models.

    Methods:
        token_ranks()
            Takes a prompt string and returns a Dict containing information about the query and a DataFrame
            with the results of the query
    """
    lm:LM

    def __init__(self, model_str:str, activations:bool = True, attention:bool = False, hidden_states:bool = True,
                 activations_layer_nums:bool=None):
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        model = AutoModelForCausalLM.from_pretrained(model_str, output_hidden_states=hidden_states, output_attentions=attention)

        lm_kwargs = {
            'collect_activations_flag': activations,
            'collect_activations_layer_nums': activations_layer_nums}
        self.lm = LM(model, tokenizer, **lm_kwargs)


    def token_ranks(self, prompt:str, token_str:str, tokens_to_generate:int = 20, layer_num:int = 0,
                    plot:bool = False, verbose:bool = False) -> Dict:
        """
        Method thatakes a prompt string and returns a Dict containing information about the query and a DataFrame
            with the results of the query
        :param prompt: The prompt to provide the model (e.g. "In move 20, White moves")
        :param token_str: A string that contains the tokens to track (e.g. " pawn rook knight bishop queen king")
            NOTE: The format of space-token is important to maintain
        :param tokens_to_generate: The number of tokens the model will generate. Default is 20
        :param layer_num: The layer to monitor for rank, numbered from output to input. Default is output (0)
        :param plot: Flag for pyplot to produce a chart
        :param verbose: Flag to print out intermediate data
        :return: Dict containing parameters and results
        """
        l:List
        legend_list:List
        l2 = []
        to_return = {"prompt":prompt, "layer_num": layer_num}

        token_dict = self.lm.tokenizer(token_str)
        tokens = token_dict['input_ids']
        output = self.lm.generate(prompt, generate=tokens_to_generate, do_sample=True, html_output=False)
        positions = np.arange(output.n_input_tokens, len(output.tokens))
        token_list = output.tokens[output.n_input_tokens:]
        legend_list = token_str.strip().split()
        for pos in positions:

            d = output.rankings_watch(watch = tokens, position = pos, printJson = False, html_output=False)
            # don't plot the graphs in output.rankings_watch
            plt.close()
            l = d['rankings']
            l2.append(l[-1])
            if verbose:
                print("\npos = {}".format(pos))
                print("input_tokens = {}\ntoken_list = {}".format(d['input_tokens'], token_list))
                print("last layer ranks = {}".format(l[-1]))

        narray = np.array(l2)
        df = pd.DataFrame(narray, columns=legend_list, index=token_list)
        to_return['data'] = df.to_dict()
        if verbose:
            print(df)
        if plot:
            df.plot(title=output.output_text, logy=True)
        return to_return

def main():
    print("Testing EccoTrendAnalytics:")
    eta = EccoTrendAnalytics("../../models/chess_model")
    d = eta.token_ranks("In move 20, White moves", " pawn rook knight bishop queen king", plot=True, verbose=True)
    print(d)
    plt.show()

if __name__ == "__main__":
    main()