Structure
=========

1. Environment [OK]
2. TD3 critic learns Q-Values in the environment [OK]
3. Gradient in TD3 critic allows to produce gradient of actions: grad_a [2]
4. A program evolves:
4.1. the program produces pi_a
4.2. we apply the TD3 gradient to map pi_a to pi*_a (pi*_a = pi_a + lr * grad_a)
4.3. we fit the program to the new actions pi*_a:
4.3.1. with an evolutionary algorithm [3] [OK]

Evolution
=========

1. DNA representation of a program [OK]
2. A cost function --> MSE [OK]

DNA: post-fix notation [1] (parse PyGAD solutions to DNA, then run DNA for evaluation)
