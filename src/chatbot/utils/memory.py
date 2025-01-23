import os
import pandas as pd
from typing import List


class Memory:
    @staticmethod
    def write_chat_history_to_file(chat_history_lst: List, file_path: str) -> None:
        """write the chat history list to a csv file

        Args:
            chat_history_lst (List[Tuple[str:, str]]): The chat history list to be written to the file.
            file_path (str): The path to the csv file where the chat history will be written.
        """

        # Create a DataFrame from the chat history list
        df = pd.DataFrame(chat_history_lst, columns=["User query", "Response"])

        # chech if the file exists and is not empty to avoid writting headers again
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            df.to_csv(file_path, mode="a", header=False, index=False, encoding="utf-8")

        else:
            df.to_csv(file_path, mode="w", header=True, index=False, encoding="utf-8")

    @staticmethod
    def read_recent_chat_history(filepath: str, num_entites: int = 2) -> List:
        try:
            recent_history = []
            last_rows = pd.read_csv(filepath).tail(num_entites)
            for _, row in last_rows.iterrows():
                row_dict = row.to_dict()
                recent_history.append(str(row_dict))

            return recent_history
        except Exception as e:
            print(f"chat history could not be loaded. {e}")
            return []
