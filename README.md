# buber
Like Uber pool for buses

What do you hate about public transport as a user? Right, waiting for the bus with an interval of 20 minutes. At winter. At -25 degrees. Or travelling full-packed bus after some unofficial event.
What do you hate about public transport from the government perspective? Empty buses travelling throughout a city leaving a carbon footprint and wasting energy. 
Why do we have these problems?
The answer is simple: static routes.
So just let's destroy the current system totally and build a new one, similar to already familiar for many citizens taxi app system. 

We propose not just dynamic intervals or smart finding of static routes: we propose a system that can build new routes effectively every day, every hour, every second! 
Key features of our solutions:
* Demand analysis on every bus stop using photos Crowd Count Neural network deployed on Azure
* Mobile app for end-users that can help them not just only find their bus to travel but also adjust the demand and make the bus arrive faster
* Server that can optimize routes using Google OP-Tools using Euristics Parallel Cheapest Insertion

Why would that work? 
* From the user perspective, you don't care which bus you take 199A or 228B, you just need to go from point A to point B the fastest way. 
* From the transporter perspective, you just need to have buses fulled, going any route. 

We used Open Data datasets to retrieve current bus stops and get information about the current state of public transport to use it as a baseline.

## Project structure

In src folder there is a source code of crowd counting NN (forked from https://github.com/svishwa/crowdcount-cascaded-mtl). 

In VRP folder there is a code of optimizing the vehicle path using Google OP-Tools 

In iOS folder you can find iOS app source code
