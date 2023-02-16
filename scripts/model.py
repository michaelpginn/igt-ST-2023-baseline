"""Defines the data model and a function to easily load data"""
from typing import Optional
import re


class IGTLine:
    """A single line of IGT"""
    def __init__(self, transcription: str, segmentation: Optional[str], glosses: str, translation: str):
        self.transcription = transcription
        self.segmentation = segmentation
        self.glosses = glosses
        self.translation = translation

    def __repr__(self):
        return f"Trnsc:\t{self.transcription}\nSegm:\t{self.segmentation}\nGloss:\t{self.glosses}\nTrnsl:\t{self.translation}\n\n"

    def gloss_list(self, segmented=False):
        """Returns the gloss line of the IGT as a list.
        :param segmented: If True, will return each morpheme gloss as a separate item.
        """
        if not segmented:
            return self.glosses.split()
        else:
            return re.split("\s|-", self.glosses)


def load_data_file(path: str) -> list[IGTLine]:
    """Loads a file containing IGT data into a list of entries."""
    all_data = []

    with open(path, 'r') as file:
        current_entry = [None, None, None, None]  # transc, segm, gloss, transl

        for line in file:
            # Determine the type of line
            # If we see a type that has already been filled for the current entry, something is wrong
            line_prefix = line[:2]
            if line_prefix == '\\t' and current_entry[0] == None:
                current_entry[0] = line[3:].strip()
            elif line_prefix == '\\m' and current_entry[1] == None:
                current_entry[1] = line[3:].strip()
            elif line_prefix == '\\g' and current_entry[2] == None:
                current_entry[2] = line[3:].strip()
            elif line_prefix == '\\l' and current_entry[3] == None:
                current_entry[3] = line[3:].strip()
                # Once we have the translation, we've reached the end and can save this entry
                all_data.append(IGTLine(transcription=current_entry[0],
                                        segmentation=current_entry[1],
                                        glosses=current_entry[2],
                                        translation=current_entry[3]))
                current_entry = [None, None, None, None]
            elif line.strip() != "":
                # Something went wrong
                print("Skipping line: ", line)
                current_entry = [None, None, None, None]
    return all_data