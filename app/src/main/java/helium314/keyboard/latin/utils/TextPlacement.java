package helium314.keyboard.latin.utils;

public final class TextPlacement {
    public String text;
    public final int startPosition;

    public TextPlacement(String text, int startPosition) {
        this.text = text;
        this.startPosition = startPosition;
    }

    public int endPosition() {
        return startPosition + text.length();
    }
}
