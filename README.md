# A Note on Generalized Second-Order Value Iteration in Markov Decision Processes  

This repository provides a Python implementation of the algorithms presented in the paper:  

**“A Note on Generalized Second-Order Value Iteration in Markov Decision Processes”** — *Journal of Optimization Theory and Applications* (2023).  
[View the paper on Springer](https://link.springer.com/article/10.1007/s10957-023-02309-x)  

---

## 📖 Overview  

Value Iteration is a classical, first-order algorithm for solving Markov Decision Processes (MDPs) by approximating the Bellman equation.  

Recently, second-order iterative methods—employing smooth approximations of the max operator—have shown interesting theoretical properties. However, they are **computationally expensive** and **numerically unstable** for large state-action spaces due to the need to evaluate exponential functions.  

To address these issues, this work introduces several **generalized first-order iterative schemes** derived from the second-order formulation. These methods:  
- Provide **global convergence guarantees**  
- Are **computationally efficient**  
- Are **easy to implement in practice**  

Theoretical comparisons and numerical simulations highlight that these schemes often **converge faster** than their second-order counterparts, while preserving theoretical rigor and enhancing practical usability.  

---

## ⚙️ Installation & Usage  

Follow these steps to set up and run the code:  

**Step 1:** Install Python MDP Toolbox  
```bash
pip install pymdptoolbox
```  

**Step 2:** Replace the default files in the MDP Toolbox with the ones from this repository  
- Replace `mdp.py` and `example.py` in the original Python MDP Toolbox with the updated versions provided here.  

**Repository Structure**  
```
├── mdp.py         # Updated MDP algorithms implementing the new iterative schemes
├── example.py     # Example usage of the proposed algorithms
├── main.py        # Script to execute experiments and generate results
└── README.md      # Project documentation
```  

**Step 3:** Run experiments  
```bash
python main.py
```  

---

## 📊 Results & Highlights  

- The proposed iterative schemes **outperform second-order methods** in terms of convergence speed in many test cases.  
- They are **computationally cheaper** and **more numerically stable**, especially in larger state-action spaces.  
- Numerical experiments confirm the theoretical results and demonstrate clear efficiency improvements.  

---

## 📜 Citation  

If you use this repository or build upon this work, please cite the paper:  

> Antony Vijesh, V., Sumithra Rudresha, S., & Abdulla, M. S. (2023). A Note on Generalized Second-Order Value Iteration in Markov Decision Processes.  
> *Journal of Optimization Theory and Applications, 199*, 1022–1049.  
> DOI: [10.1007/s10957-023-02309-x](https://doi.org/10.1007/s10957-023-02309-x)  

---

## ✨ Acknowledgements  

This work builds upon:  
- The foundational MDP algorithms in the **Python MDP Toolbox**  and **https://github.com/raghudiddigi/G-SOVI**

Special thanks to the developers of the Python MDP Toolbox for providing a robust base for this project.  
