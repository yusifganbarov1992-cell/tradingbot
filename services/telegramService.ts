export const sendTelegramMessage = async (token: string, chatId: string, message: string) => {
  if (!token || !chatId) {
    console.warn("Telegram token or Chat ID missing");
    return;
  }
  
  try {
    const url = `https://api.telegram.org/bot${token}/sendMessage`;
    await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        chat_id: chatId, 
        text: message, 
        parse_mode: 'HTML' 
      })
    });
  } catch (error) {
    console.error('Telegram Send Error', error);
  }
};

export const getTelegramChatId = async (token: string): Promise<string | null> => {
  if (!token) return null;
  try {
    // Fetches recent updates to find who messaged the bot
    const response = await fetch(`https://api.telegram.org/bot${token}/getUpdates`);
    const data = await response.json();
    
    if (data.ok && data.result && data.result.length > 0) {
      // Get the latest message's chat ID
      const lastUpdate = data.result[data.result.length - 1];
      if (lastUpdate.message && lastUpdate.message.chat) {
         return lastUpdate.message.chat.id.toString();
      }
    }
    return null;
  } catch (error) {
    console.error('Telegram Update Error', error);
    return null;
  }
};