import torch
from fuzzy_torch import FuzzyModule, FuzzyVariable, FuzzyRule, TriangularMF, TrapezoidalMF, GaussianMF
from fuzzy_torch.defuzz import Centroid

def main():
    # 1. Define Variables
    # Service: 0 to 10
    service = FuzzyVariable('Service', 0, 10)
    service.add_term('Poor', TriangularMF(0, 0, 5))
    service.add_term('Good', TriangularMF(0, 5, 10))
    service.add_term('Excellent', TriangularMF(5, 10, 10))

    # Food: 0 to 10
    food = FuzzyVariable('Food', 0, 10)
    food.add_term('Rancid', TrapezoidalMF(0, 0, 1, 3))
    food.add_term('Delicious', TrapezoidalMF(7, 9, 10, 10))

    # Tip: 0 to 30
    tip = FuzzyVariable('Tip', 0, 30)
    tip.add_term('Low', TriangularMF(0, 0, 15))
    tip.add_term('Medium', TriangularMF(0, 15, 30))
    tip.add_term('High', TriangularMF(15, 30, 30))

    # 2. Define Rules
    rules = [
        FuzzyRule(antecedent=[(service, 'Poor'), (food, 'Rancid')], consequent=(tip, 'Low')),
        FuzzyRule(antecedent=(service, 'Good'), consequent=(tip, 'Medium')),
        FuzzyRule(antecedent=[(service, 'Excellent'), (food, 'Delicious')], consequent=(tip, 'High'))
    ]

    # 3. Create System
    fis = FuzzyModule(inputs=[service, food], output=tip, rules=rules, defuzz_method=Centroid(0, 30))

    # 4. Introspection & Inference
    print("--- System Rules ---")
    fis.print_rules()
    
    # Test Case 1
    inputs = {'Service': torch.tensor([6.5]), 'Food': torch.tensor([9.8])}
    print(f"\nrunning basic check with inputs: {inputs}")
    fis.explain(inputs)

    # 5. Differentiability Check
    print("\n--- Differentiability Check ---")
    inputs_grad = {'Service': torch.tensor([6.5], requires_grad=True), 'Food': torch.tensor([9.8], requires_grad=True)}
    
    # Forward pass
    output = fis(inputs_grad)
    print(f"Output: {output.item()}")
    
    # Define a dummy loss and backward
    target = torch.tensor([25.0])
    loss = (output - target) ** 2
    loss.backward()
    
    print("Gradients on Inputs:")
    print(f"  Service Grad: {inputs_grad['Service'].grad}")
    print(f"  Food Grad: {inputs_grad['Food'].grad}")
    
    print("Gradients on Parameters (Example - Service 'Excellent' center):")
    # Access parameters of a specific MF
    p_param = service.terms['Excellent'].c
    print(f"  Service['Excellent'].c value: {p_param.item()}")
    print(f"  Service['Excellent'].c grad: {p_param.grad}")

    if p_param.grad is not None:
        print("\nSUCCESS: The system is end-to-end differentiable!")
    else:
        print("\nWARNING: Gradients not flowing to parameters.")

if __name__ == '__main__':
    main()
