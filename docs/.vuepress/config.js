module.exports = {
    title: 'AIPipeline',
    description: 'AI Pipeline',
    base: '/',
    themeConfig: {
          repo: 'FabOS-AI/fabos-base-service-ai-pipeline',
          logo: '/img/logo.svg',
          editLinks: false,
          docsDir: '',
          editLinkText: '',
          lastUpdated: false,
          nav: [
              { text: 'Home', link: '/' },
          ],

          sidebar: {
              '/docs/': [
                {
                  title: 'Getting Started',
                  collapsable: true,
                  children: [
                    'getting-started/',
                    'getting-started/overview',
                    'getting-started/installation',
                  ],
                },
                {
                  title: 'Usage',
                  collapsable: true,
                  children: [
                    'usage/',
                    'usage/pre-processing',
                    'usage/ai-toolbox',
                    'usage/evaluation'
                  ],
                },
              ]
          },
      },
      plugins: [
        '@vuepress/plugin-back-to-top',
        '@vuepress/plugin-medium-zoom',
        '@dovyp/vuepress-plugin-clipboard-copy',
        '@vuepress/medium-zoom'
      ],
}