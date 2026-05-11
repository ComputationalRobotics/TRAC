window.PAPER_SITE = {
  meta: {
    title: "TRAC",
    description:
      "TRAC: Parameter-free Optimization for Lifelong Reinforcement Learning.",
    ogImage: "assets/figures/fastTRAC.gif",
  },
  paper: {
    title:
      'Fast <strong>TRAC</strong><br><span style="font-size: 0.84em; font-weight: 400; line-height: 1.02;">A Parameter-free Optimizer for</span><br><span style="font-size: 0.84em; font-weight: 400; line-height: 1.02;">Lifelong Reinforcement Learning</span>',
    authors: [
      { name: "Aneesh Muppidi<sup>1,2</sup>", href: "https://aneeshers.github.io" },
      { name: "Zhiyu Zhang<sup>2</sup>", href: "https://zhiyuzz.github.io" },
      { name: "Heng Yang<sup>2</sup>", href: "https://hankyang.seas.harvard.edu" },
    ],
    links: [
      { label: "ArXiv", href: "https://arxiv.org/abs/2405.16642", icon: "assets/icons/arxiv-square.svg" },
      { label: "Experiments", href: "https://github.com/ComputationalRobotics/PACE?tab=readme-ov-file", icon: "assets/icons/github.png" },
      { label: "Colab", href: "https://colab.research.google.com/drive/1c5OxMa5fiSVnl5w6J7flrjNUteUkp6BV?usp=sharing", icon: "assets/icons/colab.png" },
      { label: "Pypi", href: "https://pypi.org/project/trac-optimizer/", icon: "assets/icons/python.png" },
    ],
    affiliations: "<sup>1</sup> Harvard College &nbsp;&nbsp; <sup>2</sup> Harvard SEAS",
    logo: {
      src: "assets/icons/harvard-seas.png",
      alt: "Harvard SEAS logo",
    },
    notice: "Accepted to NeurIPS 2024",
    noticeSecondary:
      'TRAC was also accepted to <strong>RLC</strong> <span style="color: var(--accent);">(Spotlight)</span>, <strong>RSS</strong> <span style="color: var(--accent);">(Spotlight)</span>, and <strong>TTIC</strong> workshops.',
    heroFigure: {
      src: "assets/figures/fastTRAC.gif",
      alt: "Fast TRAC animation",
    },
    abstract:
      "A key challenge in lifelong reinforcement learning (RL) is the loss of plasticity, where previous learning progress hinders an agent's adaptation to new tasks. While regularization and resetting can help, they require precise hyperparameter selection at the outset and environment-dependent adjustments. Building on the principled theory of online convex optimization, we present a parameter-free optimizer for lifelong RL, called TRAC, which requires no tuning or prior knowledge about the distribution shifts. Extensive experiments on Procgen, Atari, and Gym Control environments show that TRAC works surprisingly well, mitigating loss of plasticity and rapidly adapting to challenging distribution shifts, despite the underlying optimization problem being nonconvex and nonstationary.",
  },
  highlight:
    'Try <strong>TRAC</strong> in your lifelong or continual experiments with just <strong><a href="https://github.com/ComputationalRobotics/PACE">one line change.</a></strong>',
    sections: [
    {
      id: "loss-of-plasticity",
      title: "Lifelong RL suffers from <strong>Loss of Plasticity</strong>",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "In lifelong RL, a learning agent must continually acquire new knowledge to handle the nonstationarity of the environment. At first glance, there appears to be an obvious solution: given a policy gradient oracle, the agent could just keep running gradient descent nonstop. However, recent experiments have demonstrated an intriguing behavior called loss of plasticity: despite persistent gradient steps, such an agent can gradually lose its responsiveness to incoming observations.",
          ],
        },
        {
          type: "figureGrid",
          columns: 2,
          items: [
            {
              src: "assets/figures/starpilot_rewards_animation.gif",
              alt: "StarPilot rewards animation",
              caption: "Loss of plasticity appears in StarPilot.",
            },
            {
              src: "assets/figures/control_rewards_animation.gif",
              alt: "Control rewards animation",
              caption: "The same pattern appears in control environments.",
            },
          ],
        },
      ],
    },
    {
      id: "convex-help",
      title: "Surprisingly, in this non-convex setting, online <em><strong>convex</strong></em> optimization can help.",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "<strong>TRAC</strong> combines three parameter-free Online Convex Optimization (OCO) techniques: direction-magnitude decomposition, additive aggregation, and the erfi potential function. The algorithm starts with a base optimizer, <code>Base</code>, and adjusts a scaling parameter, <code>S<sub>t+1</sub></code>, in an online data-dependent manner. This parameter affects the update of <code>&theta;<sub>t+1</sub></code> as shown below.",
          ],
        },
        {
          type: "equation",
          tex: String.raw`\theta_{t+1} = S_{t+1} \cdot \theta_{t+1}^\text{base} + (1 - S_{t+1}) \theta_\text{ref}`,
          note: "The tuner uses the erfi function to adapt the scaling parameter based on incoming statistics.",
        },
        {
          type: "equation",
          tex: String.raw`s_{t+1} = \frac{\epsilon}{(\text{erfi})(1/\sqrt{2})} (\text{erfi})\left(\frac{\sigma_t}{\sqrt{2v_t} + \epsilon}\right)`,
          note: "Aggregating tuners with different discount factors allows TRAC to adaptively scale without manual tuning.",
        },
        {
          type: "figure",
          src: "assets/figures/fastTRAC.gif",
          alt: "Fast TRAC animation",
          caption: "TRAC in action on lifelong RL experiments.",
        },
      ],
    },
    {
      id: "results",
      title: "Experimental Results",
      blocks: [
        {
          type: "figure",
          src: "assets/figures/combined_plots.png",
          alt: "Combined TRAC results",
          caption: "Main results across Procgen, Atari, and Gym Control.",
        },
      ],
    },
    {
      id: "usage",
      title: "Try <strong>TRAC</strong> in PyTorch with <em>one line</em>",
      blocks: [
        {
          type: "callout",
          tone: "yellow",
          html:
            'For full examples using <strong>TRAC</strong> with PPO in lifelong RL see <a href="https://github.com/ComputationalRobotics/TRAC/tree/main">here</a>.',
        },
        {
          type: "prose",
          paragraphs: [
            "<strong>First Install TRAC</strong> <a href=\"https://pypi.org/project/trac-optimizer/\">[Pypi]</a>",
          ],
        },
        {
          type: "code",
          language: "bash",
          code: String.raw`pip install trac-optimizer`,
        },
        {
          type: "prose",
          paragraphs: [
            "Original",
          ],
        },
        {
          type: "code",
          language: "python",
          code: String.raw`from torch.optim import Adam
# original code
optimizer = Adam(model.parameters(), lr=0.01)
# your typical optimizer methods
optimizer.zero_grad()
optimizer.step()`,
        },
        {
          type: "prose",
          paragraphs: [
            "With TRAC",
          ],
        },
        {
          type: "code",
          language: "python",
          code: String.raw`from trac_optimizer import start_trac
# with TRAC
optimizer = start_trac(log_file='logs/trac.text', Adam)(model.parameters(), lr=0.01)
# using your optimizer methods exactly as you did before (feel free to use others as well)
optimizer.zero_grad()
optimizer.step()`,
        },
      ],
    },
    {
      id: "acknowledgements",
      title: "Acknowledgements",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            'We thank <a href="https://ashok.cutkosky.com">Ashok Cutkosky</a> for insightful discussions on online optimization in nonstationary settings. We are grateful to <a href="https://david-abel.github.io">David Abel</a> for his thoughtful insights on loss of plasticity in relation to lifelong reinforcement learning. We appreciate <a href="https://kzhang66.github.io">Kaiqing Zhang</a> and <a href="https://huyangsh.github.io">Yang Hu</a> for their comments on theoretical and nonstationary RL. This project is partially funded by <a href="https://research.fas.harvard.edu/deans-competitive-fund-promising-scholarship#:~:text=This%20program%20is%20open%20to,not%20eligible%20for%20this%20program.">Harvard University Dean\'s Competitive Fund for Promising Scholarship.</a>',
          ],
        },
      ],
    },
  ],
  footer: {
    left: "TRAC project page built from the minimal paper template.",
    right:
      '<a href="https://github.com/Aneeshers/research-paper">TRAC project page built from the minimal paper template</a>',
  },
};
