# cellular-automata

A simple program for runing simulations entirely on the GPU.

Each simulation is just 2 shaders: 1 for logic and 1 for rendering.

On an RTX 4090 it can easily do over 6000 ticks per second.
Even integrated graphics can run 240 ticks per second without dropping frames.

![Example screenshot](example.png)

# Two Paradox
so you would find that most simulation reach some sort of steady state. 
however for two-paradox which is a specific rock paper scissors like game. 
that steady state is qualitativly diffrent.

The reason for it is that no local area can remain stable to outside forces. 
This is because every period of 3 colors swaping has at least 1 intresting external interaction.
When compared to the classic extension of rock paper scissors to 7 members the diffrence is stark.
That version is more likely to set into a quicker period.