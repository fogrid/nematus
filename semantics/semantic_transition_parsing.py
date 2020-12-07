"""
# HOW TO USE
1. Run Tupa TODO
2. Change parameters TODO
3. Run to get the transitions text
"""
import os
import re
from collections import deque

from ucca.core import Passage
from ucca.ioutil import read_files_and_dirs

from tupa.action import Actions
from tupa.oracle import Oracle
from tupa.states.node import Node
from tupa.states.state import State
from sys import argv

DEFAULT_INPUT = os.path.join("data", "newstest2012.unesc.en_ucca_res")
DEFAULT_OUTPUT = "newstest2012.unesc.en_ucca_transitions.txt"
DEFAULT_BPE_PATH = os.path.join("data", "newstest2012.unesc.tok.tc.bpe.en")

BPE_SEP = "@@"

NODE_ACT_MARK = "##|"
NODE_WORD_MARK = "%%|"
EDGE_MARK = "@@|"
REMOTE_EDGE_MARK = EDGE_MARK  # treating remote edges as edges with special label
STACK_ACTION_MARK = "$$|"

BAD_LINE_STR = "===BAD LINE==="

def custom_node_str(node, **kwargs):
    s = '%s' % node.text if node.text else node.node_id + NODE_WORD_MARK  or str(node.index) + NODE_WORD_MARK
    if node.label:
        s += "/" + node.label
    return s

Node.__str__ = custom_node_str

def xml_to_tranisions(source_ucca_dir, output_path, bpe_path, lines_file_path):
    ucca_file_list = sorted(os.listdir(source_ucca_dir), key=lambda f: int(re.sub('\D', '', f)))
    ucca_file_list = [os.path.join(source_ucca_dir, fname) for fname in ucca_file_list]
    all_ucca_passages = read_files_and_dirs(ucca_file_list)

    with open(output_path, "w") as output_file:
        with open(lines_file_path, "w+") as lines_file:
            with open(bpe_path, "r") as bpe_file:
                for i, (passage, bpe_passage) in enumerate(zip(all_ucca_passages, bpe_file)):
                    print(f"passage {i}: {str(passage)}")
                    bpe_passage = bpe_passage.replace("``", "` `") # UCCA treats `` as two words, bpe as one
                    print(f"bpe passage: {bpe_passage}")
                    try:
                        transition_str = ucca_passage_to_transition_line(passage, bpe_passage)
                        # print(f"\nline: {transition_str}")
                        print(transition_str, file=output_file)
                        print(i, file=lines_file)
                    except Exception as e:
                        print(f"failed when trying to parse line {i}, writing \"{BAD_LINE_STR}\".\n Excpetion: {e}")
                        print(BAD_LINE_STR, file=output_file)
                        continue

    print("done!")

def ucca_passage_to_transition_line(passage: Passage, bpe_passage: str):
    oracle = Oracle(passage)
    state = State(passage)
    actions = Actions()
    line = "" # "root" + ROOT_MARK + " "
    raw_line = ""
    num_to_skip_in_buffer = 0

    bpe_iter = iter(bpe_passage.split())
    while not state.finished:
        try:
            action = min(oracle.get_actions(state, actions).values(), key=str)
        except AssertionError as e:
            print(e)
            print(f"bpe line : \"{bpe_passage}\"" +
                  f"stack: {state.stack}\n" +
                  f"buffer: {state.buffer}\n" +
                  f"full ucca passage: {passage}")
            raise e
        state.transition(action)
        s = str(action)
        print(f"act={s}", end=", ")
        if state.need_label:

            label, _ = oracle.get_label(state, action)
            state.label_node(label)
            s += "_" + str(label)
        raw_s = s
        raw_line += raw_s + " "
        print(f"initial s={s}", end=", ")
        if action.is_type(Actions.Shift):
            if not num_to_skip_in_buffer:
                # add the word, split using the BPE format (it's->it@@'s)
                try:
                    s = ""
                    num_sep_chars = 0
                    for token in bpe_iter:
                        s += token

                        if not token.endswith(BPE_SEP):
                            break
                        s += " "
                        num_sep_chars += 3 # 2 for @@, 1 for space
                    print(f"adding word. ucca word: \"{state.stack[-1]}\"  bpe word:\"{s}\" ")
                    # some tokens are interpreted as two tokens by UCCA, but one word by the bpe. so drop the
                    # extra tokens from the graph.
                    actual_bpe_word_len = len(s) - num_sep_chars
                    ucca_word_len = len(str(state.stack[-1]))
                    while actual_bpe_word_len > ucca_word_len:
                        print(f"num_of_sep_chars={num_sep_chars}, actual_bpe_word_len={actual_bpe_word_len}, "
                              f"ucca_word_len={ucca_word_len}")
                        print(f"dropping extra tokens from stack. s={s}, popping word:{state.buffer[0]}")
                        raw_line += str(Actions.Shift) + STACK_ACTION_MARK + " " + \
                                    str(Actions.Reduce) + STACK_ACTION_MARK + " "
                        ucca_word_len += len(str(state.buffer[0]))
                        state.transition(Actions.Shift)
                        state.transition(Actions.Reduce)
                    assert s, f"Couldn't take word from bpe file. Maybe it's not fitting the input file? \n" \
                              f"curr line: \"{line}\"\n " \
                              f"bpe line : \"{bpe_passage}\"" \
                              f"stack: {state.stack}\n" \
                              f"buffer: {state.buffer}\n" \
                              f"full ucca passage: {passage}"
                except AssertionError as e:
                    print(e)
                    raise e
            else:  # we already added this word so we don't want to re-add it, or it's a node -
                #   just add the shift action.
                num_to_skip_in_buffer -= 1
                s += STACK_ACTION_MARK

        elif action.is_type(Actions.Reduce):
            s += STACK_ACTION_MARK
        elif action.is_type(Actions.Swap):
            num_to_skip_in_buffer += 1
            s += STACK_ACTION_MARK
        elif action.is_type(Actions.Node):  # creating a new node
            if str(action).endswith("Terminal"):  # no need to actually add nodes for terminals,
                                                 # just add the word
                raw_line +=  str(Actions.Reduce)  + STACK_ACTION_MARK + " " + \
                             str(Actions.Shift) + STACK_ACTION_MARK + " "
                print(f"doing reduce-shift without writing")
                assert min(oracle.get_actions(state, actions).values(), key=str) == Actions.Reduce
                state.transition(Actions.Reduce)
                assert min(oracle.get_actions(state, actions).values(), key=str) == Actions.Shift
                state.transition(Actions.Shift)
                continue
            else:
                s += NODE_ACT_MARK
                num_to_skip_in_buffer += 1
        elif action.is_type(Actions.RightRemote) or action.is_type(Actions.RightRemote):  # is edge
            s += REMOTE_EDGE_MARK
        elif action.is_type(Actions.Finish):  # FINISH is always happening, no point in actually adding it
            continue
        else:  # is edge - also for root
            s += EDGE_MARK
        print(f"final s={s}")
        line += s + " "
    print(f"raw_line={raw_line}")
    print(f"final_line={line}")
    print()
    return line


def main():
    if len(argv) == 4:
        print(f"using cmd args")
        source_dir, output_path, bpe_path = argv[1:]
    else:
        print(f"using default args")
        source_dir = DEFAULT_INPUT
        output_path = DEFAULT_OUTPUT
        bpe_path = DEFAULT_BPE_PATH
    lines_file_path = output_path + ".line_nums"
    print(f"args: source_dir={source_dir}, output_path={output_path}, bpe_path={bpe_path}, writing line num to: {lines_file_path}")
    xml_to_tranisions(source_dir, output_path, bpe_path, lines_file_path)


if __name__ == "__main__":
    main()