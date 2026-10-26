/**
 * 前端主题样式与全局配色管理模块 (Theme Management).
 *
 * 统一应用暖色调护眼浅色主题，覆盖系统或本地历史配置，
 * 确保心理辅导与后台教师工作台视觉风格一致且平和温暖。
 */

/**
 * 应用全局统一浅色主题调色板并重置浏览器 meta 标签颜色。
 */
export function applyTheme(): void {
  document.documentElement.dataset.theme = 'light'
  document.documentElement.style.colorScheme = 'light'
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', '#f6f5ee')
  try {
    localStorage.removeItem('solaceai.theme')
  } catch {
    // 即使在无本地存储访问权限的受限环境（如隐身模式）下，主题配色方案依然正常生效
  }
}
