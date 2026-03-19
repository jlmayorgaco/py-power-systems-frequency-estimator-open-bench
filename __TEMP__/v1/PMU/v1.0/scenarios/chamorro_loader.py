import pandas as pd
import numpy as np


class ChamorroLoader:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)

    def process(self):
        """
        Retorna t, v_input (monofásico equivalente), f_ref
        """
        # 1. Extraer tiempo
        t = self.df["t"].values

        # 2. Lógica de Voltajes (Clarke Transform)
        # Si el CSV tiene va, vb, vc, sacamos la componente Alpha (monofásica rica)
        if {"va", "vb", "vc"}.issubset(self.df.columns):
            va = self.df["va"].values
            vb = self.df["vb"].values
            vc = self.df["vc"].values
            # Alpha = (2/3)*(va - 0.5*vb - 0.5*vc)
            v_input = (2.0 / 3.0) * (va - 0.5 * vb - 0.5 * vc)
        else:
            # Si solo hay una fase, usamos 'va' o la primera columna de voltaje
            v_input = (
                self.df["va"].values
                if "va" in self.df.columns
                else self.df.iloc[:, 1].values
            )

        # 3. Frecuencia de referencia (Simulink o PMU)
        f_ref = self.df["f"].values if "f" in self.df.columns else None

        return t, v_input, f_ref


# --- TEST RÁPIDO PARA CUANDO TENGAS EL CSV ---
# loader = ChamorroLoader("evento_real.csv")
# t, v, f = loader.process()
